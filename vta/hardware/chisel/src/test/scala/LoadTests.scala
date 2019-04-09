// See LICENSE.txt for license details.
package vta

import chisel3._
import chisel3.util._
import chisel3.testers._
import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}

class LoadTests(c: Load)(implicit val p: freechips.rocketchip.config.Parameters)
    extends PeekPokeTester(c) {

  val insn0 = "h00000010000400040000000000000080".U // load - wgt load
  val insn1 = "h00000010000400080000000000000140".U // load - inp load
  val insns = IndexedSeq(insn0, insn1)

  val input0 = "h1e29f90808f2edf12907d7cd3914213d".U
  val input1 = "hef3f052a0d1fcc383c17d7f6e2ebe628".U
  val input2 = "h1febe4030bedd5c7cf0e25153ced2912".U
  val input3 = "h20fb3cf33621c914f329cfc8c4240c36".U
  val inputs = Vector(input0, input1, input2, input3)

  val valid_counts = List(16, 32)

  def test_load_weights(){

  // reset
  step(1)

  // write instruction
  poke(c.io.load_queue.data, insns(0))
  poke(c.io.load_queue.valid, 1.U)
  step(1)
  expect(c.io.load_queue.ready, 1.U)
  poke(c.io.weights.waitrequest, 1.U)
  step(1)
  expect(c.io.load_queue.ready, 0.U)
  step(1)

  for (j <- 0 to valid_counts(0) - 1){

  expect(c.io.weights.address, j << 4)
  poke(c.io.weights.readdata, 0.U)
  poke(c.io.weights.waitrequest, 1.U)
  poke(c.io.load_queue.valid, 0.U)
  poke(c.io.load_queue.data, 0.U)
  step(1)
  expect(c.io.weights.address, j << 4)
  expect(c.io.weights.read, 1.U)
  poke(c.io.weights.waitrequest, 0.U)
  poke(c.io.weights.readdata, inputs(j % 4))
  step(1)

  } // end of for loop

  poke(c.io.inputs.readdata, 0.U)
  poke(c.io.weights.readdata, 0.U)
  step(1)
  step(1)
  step(1)

  step(1)
  step(1)
  step(1)

  } // end of function

  def test_load_inputs(){

  // reset
  step(1)

  // write instruction
  poke(c.io.load_queue.data, insns(1))
  poke(c.io.load_queue.valid, 1.U)
  step(1)
  expect(c.io.load_queue.ready, 1.U)
  poke(c.io.inputs.waitrequest, 1.U)
  step(1)
  expect(c.io.load_queue.ready, 0.U)
  step(1)

  for (j <- 0 to valid_counts(1) - 1){

  expect(c.io.inputs.address, j << 4)
  poke(c.io.inputs.readdata, 0.U)
  poke(c.io.inputs.waitrequest, 1.U)
  poke(c.io.load_queue.valid, 0.U)
  poke(c.io.load_queue.data, 0.U)
  step(1)
  expect(c.io.inputs.address, j << 4)
  expect(c.io.inputs.read, 1.U)
  poke(c.io.inputs.waitrequest, 0.U)
  poke(c.io.inputs.readdata, inputs(j % 4))
  step(1)

  } // end of for loop

  // dependency queue processing
  step(1)
  step(1)
  expect(c.io.l2g_dep_queue.valid, 1.U)
  poke(c.io.inputs.readdata, 0.U)
  poke(c.io.weights.readdata, 0.U)
  poke(c.io.l2g_dep_queue.ready, 1.U)
  step(1)
  poke(c.io.l2g_dep_queue.ready, 0.U)
  step(1)
  expect(c.io.l2g_dep_queue.valid, 0.U)
  step(1)
  step(1)
  step(1)

  } // end of function

  test_load_weights()
  test_load_inputs()
}

class LoadTester extends ChiselFlatSpec {
  implicit val p = (new VTAConfig).toInstance
  behavior of "Load"
  backends foreach {backend =>
    it should s"perform correct math operation on dynamic operand in $backend" in {
      Driver(() => new Load(), backend)((c) => new LoadTests(c)) should be (true)
    }
  }
}

