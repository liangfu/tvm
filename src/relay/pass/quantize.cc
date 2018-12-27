/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file quantize.cc
 *
 * \brief transform a graph to a low-bit graph
 *   for compression and acceleration.
 */
#include <cmath>
#include <string>
#include <stack>
#include <dmlc/thread_local.h>
#include <tvm/base.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include "./quantize.h"
#include "./pattern_util.h"


namespace tvm {
namespace relay {
namespace quantize {

/*! \brief Attribute for simulated quantize operator */
struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  bool sign;
  std::string rounding;
  int kind;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(sign).set_default(true);
    TVM_ATTR_FIELD(rounding).set_default("round")
        .describe("rounding mode. Can be 'floor', 'ceil', 'round'");
    TVM_ATTR_FIELD(kind)
        .describe("kind of field.");
  }
};

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

bool SimulatedQuantizeRel(const Array<Type>& types,
                          int num_inputs,
                          const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 5);
  const auto param = attrs.as<SimulatedQuantizeAttrs>();
  CHECK(param != nullptr);

  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";

  reporter->Assign(types[1], TensorTypeNode::make({}, Float(32)));    // dom_scale
  reporter->Assign(types[2], TensorTypeNode::make({}, Float(32)));    // clip_min
  reporter->Assign(types[3], TensorTypeNode::make({}, Float(32)));    // clip_max
  reporter->Assign(types[4], types[0]);                               // output
  return true;
}

RELAY_REGISTER_OP("simulated_quantize")
.describe(R"code(simulated quantize op)code" TVM_ADD_FILELINE)
.set_num_inputs(4)
.add_argument("data", "Tensor", "The input data.")
.add_argument("dom_scale", "Tensor", "The domain scale of input data. It should be a scalar")
.add_argument("clip_min", "Tensor", "lower bound. It should be a scalar")
.add_argument("clip_max", "Tensor", "upper bound. It should be a scalar")
.set_attrs_type_key("relay.attrs.SimulatedQuantizeAttrs")
.set_support_level(10)
.add_type_rel("SimulatedQuantize", SimulatedQuantizeRel);

TVM_REGISTER_API("relay._quantize.simulated_quantize")
.set_body_typed<Expr(Expr, Expr, Expr, Expr, bool, std::string, int)>(
  [](Expr data, Expr dom_scale, Expr clip_min, Expr clip_max,
     bool sign, std::string rounding, int kind) {
    auto attrs = make_node<SimulatedQuantizeAttrs>();
    attrs->sign = sign;
    attrs->rounding = rounding;
    attrs->kind = kind;
    static const Op& op = Op::Get("simulated_quantize");
    return CallNode::make(op, {data, dom_scale, clip_min, clip_max}, Attrs(attrs), {});
  });


// =============
// annotate pass

Expr QAnnotateExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  if (cfg->store_lowbit_output) {
    // store low bit output back for VTA
    const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
    return (*f)(this->expr, static_cast<int>(kQInput));
  } else {
    return expr;
  }
}

QAnnotateExpr QAnnotateExprNode::make(Expr expr, QAnnotateKind kind) {
  auto rnode = make_node<QAnnotateExprNode>();
  rnode->expr = expr;
  rnode->kind = kind;
  return QAnnotateExpr(rnode);
}

TVM_REGISTER_API("relay._quantize.make_annotate_expr")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    *ret = QAnnotateExprNode::make(args[0],
      static_cast<QAnnotateKind>(args[1].operator int()));
  });


TVM_REGISTER_API("relay._quantize.annotate")
.set_body_typed<Expr(Expr)>([] (const Expr& expr) {
  std::function<Expr(const Expr&)> fmulti_ref = [](const Expr& e) {
      if (e->derived_from<TempExprNode>()) {
        const auto* n = e.as<QAnnotateExprNode>();
        CHECK(n);
        const PackedFunc* f = runtime::Registry::Get("relay.quantize.attach_simulated_quantize");
        Expr ret = (*f)(n->expr, static_cast<int>(kQInput));
        return static_cast<Expr>(QAnnotateExprNode::make(ret, kQInput));
      }
      return e;
    };
  return ForwardRewrite(expr, "FQAnnotateRewrite", nullptr, fmulti_ref);
});


// =============
// realize pass

Expr QRealizeIntExprNode::Realize() const {
  const auto& cfg = QConfig::Current();
  Expr data = this->data;
  if (cfg->store_lowbit_output) {
    data = Cast(data, cfg->dtype_input);
  }
  // dequantize
  data = Cast(data, Float(32));
  data = Multiply(data, this->dom_scale);
  return data;
}

QRealizeIntExpr QRealizeIntExprNode::make(Expr data, Expr dom_scale, DataType dtype) {
  NodePtr<QRealizeIntExprNode> n = make_node<QRealizeIntExprNode>();
  n->data = std::move(data);
  n->dom_scale = std::move(dom_scale);
  n->dtype = std::move(dtype);
  return QRealizeIntExpr(n);
}


inline Expr ForwardOp(const Call& ref_call, const Array<Expr>& args) {
  return CallNode::make(ref_call->op,
    args, ref_call->attrs, ref_call->type_args);
}


/* calculate `data * s1 / s2`, use shift if possible */
inline Expr MulAndDiv(Expr data, float s1, float s2) {
  const QConfig& cfg = QConfig::Current();
  float shift_factor = std::log2(s1 / s2);
  if (static_cast<int>(shift_factor) == shift_factor) {
    return LeftShift(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(shift_factor)));
  } else {
    data = Cast(data, Float(32));
    return Multiply(data, MakeConstantScalar(Float(32), s1 / s2));
  }
}

Expr QuantizeRealize(const Call& ref_call,
                     const Array<Expr>& new_args,
                     const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  // do not handle data type cast
  const auto param = ref_call->attrs.as<SimulatedQuantizeAttrs>();
  CHECK_EQ(param->rounding, "round");

  Expr dom_scale = new_args[1];
  Expr clip_min = new_args[2];
  Expr clip_max = new_args[3];

  float dom_scale_imm = GetScalarFromConstant<float>(dom_scale);
  float clip_min_imm = GetScalarFromConstant<float>(clip_min);
  float clip_max_imm = GetScalarFromConstant<float>(clip_max);

  // x * idom_scale = y * odom_scale
  // => y = x * idom_scale / odom_scale
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr data = n->data;
    float idom_scale_imm = GetScalarFromConstant<float>(n->dom_scale);
    float odom_scale_imm = GetScalarFromConstant<float>(dom_scale);
    float shift_nbit = std::log2(odom_scale_imm / idom_scale_imm);
    // int32->int8
    if (static_cast<int>(shift_nbit) == shift_nbit) {
      // use shift
      if (cfg->round_for_shift) {
        float round_bias = std::pow(2, shift_nbit - 1);
        data = Add(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(round_bias)));
      }
      data = RightShift(data, MakeConstantScalar(cfg->dtype_activation, static_cast<int>(shift_nbit)));
      data = Clip(data, clip_min_imm, clip_max_imm);
      return QRealizeIntExprNode::make(data, dom_scale, n->dtype);
    } else {
      // float computation
      data = Cast(data, Float(32));
      Expr scaled_data = Multiply(data, Divide(n->dom_scale, dom_scale));
      Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
      return QRealizeIntExprNode::make(round_data, dom_scale, Float(32));
    }
  }

  // quantize from real
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  Expr data = new_args[0];
  Expr scaled_data = Multiply(data, MakeConstantScalar(Float(32), 1 / dom_scale_imm));
  Expr round_data = Clip(Round(scaled_data), clip_min_imm, clip_max_imm);
  return QRealizeIntExprNode::make(round_data, dom_scale, Float(32));
}

RELAY_REGISTER_OP("simulated_quantize")
.set_attr<FForwardRewrite>("FQRealizeRewrite", QuantizeRealize);


Expr Conv2dRealize(const Call& ref_call,
                   const Array<Expr>& new_args,
                   const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>()) {
    return Expr(nullptr);
  }
  const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
  CHECK(lhs);
  const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
  CHECK(rhs);

  Expr ldata = Cast(lhs->data, cfg->dtype_input);
  Expr rdata = Cast(rhs->data, cfg->dtype_weight);

  const auto ref_attrs = ref_call->attrs.as<Conv2DAttrs>();
  auto attrs = make_node<Conv2DAttrs>();
  *attrs = *ref_attrs;
  DataType out_dtype = cfg->dtype_activation;
  attrs->out_dtype = out_dtype;

  Expr ret = CallNode::make(ref_call->op,
    {ldata, rdata}, Attrs(attrs), ref_call->type_args);
  Expr dom_scale = FoldConstant(Multiply(lhs->dom_scale, rhs->dom_scale));
  return QRealizeIntExprNode::make(ret, dom_scale, out_dtype);
}

RELAY_REGISTER_OP("nn.conv2d")
.set_attr<FForwardRewrite>("FQRealizeRewrite", Conv2dRealize);


Expr MulRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype == Float(32)) {
      ldata = Cast(ldata, dtype);
    } else {
      CHECK_EQ(lhs->dtype, dtype);
    }
    if (rhs->dtype == Float(32)) {
      rdata = Cast(rdata, dtype);
    } else {
      CHECK_EQ(rhs->dtype, dtype);
    }

    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    Expr dom_scale = FoldConstant(Multiply(lhs->dom_scale, rhs->dom_scale));
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("multiply")
.set_attr<FForwardRewrite>("FQRealizeRewrite", MulRealize);


Expr AddRealize(const Call& ref_call,
                const Array<Expr>& new_args,
                const NodeRef& ctx) {
  const QConfig& cfg = QConfig::Current();
  CHECK_EQ(new_args.size(), 2);
  if (new_args[0].as<QRealizeIntExprNode>() && new_args[1].as<QRealizeIntExprNode>()) {
    const auto* lhs = new_args[0].as<QRealizeIntExprNode>();
    const auto* rhs = new_args[1].as<QRealizeIntExprNode>();
    Expr ldata = lhs->data;
    Expr rdata = rhs->data;

    // unify the data type
    DataType dtype = cfg->dtype_activation;
    if (lhs->dtype == Float(32)) {
      ldata = Cast(ldata, dtype);
    } else {
      CHECK_EQ(lhs->dtype, dtype);
    }
    if (rhs->dtype == Float(32)) {
      rdata = Cast(rdata, dtype);
    } else {
      CHECK_EQ(rhs->dtype, dtype);
    }

    // unify the dom_scale
    // x = a * s1, y = b * s2
    // x + y = (a * s1 / s2 + b) * s2, if s1 > s2
    //       = (a + b * s2 / s1) * s1, if s2 > s1
    Expr dom_scale;
    float s1 = GetScalarFromConstant<float>(lhs->dom_scale);
    float s2 = GetScalarFromConstant<float>(rhs->dom_scale);
    if (s1 > s2) {
      dom_scale = rhs->dom_scale;
      ldata = MulAndDiv(ldata, s1, s2);
    } else if (s1 < s2) {
      dom_scale = lhs->dom_scale;
      rdata = MulAndDiv(rdata, s2, s1);
    } else {
      dom_scale = lhs->dom_scale;
    }
    Expr ret = ForwardOp(ref_call, {ldata, rdata});
    return QRealizeIntExprNode::make(ret, dom_scale, dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>() && !new_args[1]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("add")
.set_attr<FForwardRewrite>("FQRealizeRewrite", AddRealize);


Expr ReluRealize(const Call& ref_call,
                 const Array<Expr>& new_args,
                 const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  if (const auto* n = new_args[0].as<QRealizeIntExprNode>()) {
    Expr ret = ForwardOp(ref_call, {n->data});
    return QRealizeIntExprNode::make(ret, n->dom_scale, n->dtype);
  }
  CHECK(!new_args[0]->derived_from<TempExprNode>());
  return Expr(nullptr);
}

RELAY_REGISTER_OP("nn.relu")
.set_attr<FForwardRewrite>("FQRealizeRewrite", ReluRealize);


TVM_REGISTER_API("relay._quantize.realize")
.set_body_typed<Expr(Expr)>([](const Expr& e) {
  Expr ret = ForwardRewrite(e, "FQRealizeRewrite", nullptr, nullptr);
  return ret;
});


// =============
// qconfig

QConfig qconfig() {
  return QConfig(make_node<QConfigNode>());
}

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMQConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  QConfig default_config;

  /*! \brief The current build config context */
  std::stack<QConfig> context_stack;

  TVMQConfigThreadLocalEntry() :
    default_config(qconfig()) {
  }
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMQConfigThreadLocalEntry> TVMQConfigThreadLocalStore;

void QConfig::EnterQConfigScope(const QConfig& build_config) {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void QConfig::ExitQConfigScope() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

QConfig QConfig::Current() {
  TVMQConfigThreadLocalEntry *entry = TVMQConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(QConfigNode);

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<QConfigNode>([](const QConfigNode *op, IRPrinter *p) {
  p->stream << "qconfig(";
  p->stream << "nbit_input=" << op->nbit_input << ", ";
  p->stream << "nbit_weight=" << op->nbit_weight << ", ";
  p->stream << "nbit_activation=" << op->nbit_activation << ", ";
  p->stream << "global_scale=" << op->global_scale << ", ";
  p->stream << "skip_k_conv==" << op->skip_k_conv << ", ";
  p->stream << "round_for_shift==" << op->round_for_shift << ", ";
  p->stream << "store_lowbit_output==" << op->store_lowbit_output;
  p->stream << ")";
});

TVM_REGISTER_API("relay._quantize._GetCurrentQConfig")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = QConfig::Current();
  });

TVM_REGISTER_API("relay._quantize._EnterQConfigScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  QConfig target = args[0];
  QConfig::EnterQConfigScope(target);
  });

TVM_REGISTER_API("relay._quantize._ExitQConfigScope")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  QConfig::ExitQConfigScope();
  });

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
