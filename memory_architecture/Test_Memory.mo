model Test_Memory
  import Modelica.Blocks.Sources.Ramp;
  import Modelica.Blocks.Sources.Constant;

  Ramp dutySweep(height=0.9, duration=1, offset=0.05)
    annotation(Placement(transformation(extent={{-60,20},{-40,40}})));
  Constant errorTest(k=0.3)
    annotation(Placement(transformation(extent={{-60,-40},{-40,-20}})));

  Memory_Architecture mem
    annotation(Placement(transformation(extent={{0,-20},{40,20}})));

equation
  connect(dutySweep.y, mem.Duty_Cycle);
  connect(errorTest.y, mem.Error);

end Test_Memory;
