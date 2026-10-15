model Memory_Architecture
  import Modelica.Blocks.Interfaces.RealInput;
  import Modelica.Blocks.Interfaces.RealOutput;
  import Modelica.Blocks.Nonlinear.Limiter;
  import Modelica.Blocks.Tables.CombiTable2Ds;
  import Modelica.Blocks.Types.Smoothness;
  import Modelica.Blocks.Types.Extrapolation;

  RealInput Duty_Cycle annotation(Placement(transformation(extent={{-120,40},{-100,60}})));
  RealInput Error annotation(Placement(transformation(extent={{-120,-60},{-100,-40}})));
  RealOutput Best_Duty_Cycle annotation(Placement(transformation(extent={{100,-10},{120,10}})));

  Limiter Sat1(uMax=0.95, uMin=0.05)
    annotation(Placement(transformation(extent={{-60,30},{-20,70}})));
  Limiter Sat2(uMax=0.5, uMin=-0.5)
    annotation(Placement(transformation(extent={{-60,-70},{-20,-30}})));

  CombiTable2Ds table(
    table = [0, -0.5, 0.0, 0.5;
             0.05, 0.10, 0.20, 0.30;
             0.50, 0.40, 0.50, 0.60;
             0.95, 0.70, 0.80, 0.90],
    smoothness = Smoothness.LinearSegments,
    extrapolation = Extrapolation.HoldLastPoint)
    annotation(Placement(transformation(extent={{0,-20},{40,20}})));

equation
  connect(Duty_Cycle, Sat1.u);
  connect(Sat1.y, table.u1);

  connect(Error, Sat2.u);
  connect(Sat2.y, table.u2);

  connect(table.y, Best_Duty_Cycle);

end Memory_Architecture;
