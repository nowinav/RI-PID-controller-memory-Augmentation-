model BuckConverter_Averaged
  extends Modelica.Icons.Example;

  // Parameters
  parameter Modelica.Units.SI.Inductance L = 100e-6;
  parameter Modelica.Units.SI.Capacitance C = 100e-6;
  parameter Modelica.Units.SI.Resistance R = 10;
  parameter Modelica.Units.SI.Resistance R_esr = 1.0;

  // RL interfaces
  Modelica.Blocks.Interfaces.RealInput duty_cycle(start = 0.42)
    annotation(Placement(transformation(origin = {-100, 30}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput Vin(start = 12)
    annotation(Placement(transformation(origin = {-100, 0}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealOutput Vout
    annotation(Placement(transformation(origin = {110, 0}, extent = {{-10, -10}, {10, 10}})));

  // Averaged PWM source
  Modelica.Blocks.Math.Product product
    annotation(Placement(transformation(origin = {-50, 15}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Sources.SignalVoltage averaged_source
    annotation(Placement(transformation(origin = {-66, -40}, extent = {{10, -10}, {-10, 10}}, rotation = -90)));

  // LC filter + damping
  Modelica.Electrical.Analog.Basic.Inductor inductor(L = L)
    annotation(Placement(transformation(origin = {-10, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Resistor inductor_esr(R = R_esr)
    annotation(Placement(transformation(origin = {20, -20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor(C = C)
    annotation(Placement(transformation(origin = {50, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Electrical.Analog.Basic.Resistor resistor(R = R)
    annotation(Placement(transformation(origin = {80, -40}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Electrical.Analog.Basic.Ground ground
    annotation(Placement(transformation(origin = {20, -70}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor
    annotation(Placement(transformation(origin = {80, 20}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));

equation
  // Averaged PWM
  connect(duty_cycle, product.u1) annotation(Line(points = {{-100, 30}, {-70, 30}, {-70, 18}, {-62, 18}}, color = {0, 0, 127}));
  connect(Vin, product.u2) annotation(Line(points = {{-100, 0}, {-70, 0}, {-70, 12}, {-62, 12}}, color = {0, 0, 127}));
  connect(product.y, averaged_source.v) annotation(Line(points = {{-39, 15}, {-30, 15}, {-30, -40}, {-54, -40}}, color = {0, 0, 127}));

  // Power stage
  connect(averaged_source.p, inductor.p) annotation(Line(points = {{-66, -50}, {-66, -20}, {-20, -20}}, color = {0, 0, 255}));
  connect(inductor.n, inductor_esr.p) annotation(Line(points = {{0, -20}, {10, -20}}, color = {0, 0, 255}));
  connect(inductor_esr.n, capacitor.p) annotation(Line(points = {{30, -20}, {50, -20}, {50, -30}}, color = {0, 0, 255}));
  connect(inductor_esr.n, resistor.p) annotation(Line(points = {{30, -20}, {80, -20}, {80, -30}}, color = {0, 0, 255}));
  connect(capacitor.n, ground.p) annotation(Line(points = {{50, -50}, {50, -60}, {20, -60}}, color = {0, 0, 255}));
  connect(resistor.n, ground.p) annotation(Line(points = {{80, -50}, {80, -60}, {20, -60}}, color = {0, 0, 255}));
  connect(averaged_source.n, ground.p) annotation(Line(points = {{-66, -30}, {-66, -60}, {20, -60}}, color = {0, 0, 255}));

  // Sensor
  connect(voltageSensor.p, inductor_esr.n) annotation(Line(points = {{80, 10}, {80, -20}, {30, -20}}, color = {0, 0, 255}));
  connect(voltageSensor.n, ground.p) annotation(Line(points = {{80, 30}, {80, 40}, {40, 40}, {40, -60}, {20, -60}}, color = {0, 0, 255}));
  connect(voltageSensor.v, Vout) annotation(Line(points = {{90, 20}, {100, 20}, {100, 0}, {110, 0}}, color = {0, 0, 127}));

end BuckConverter_Averaged;