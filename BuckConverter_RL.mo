model BuckConverter_Averaged
  extends Modelica.Icons.Example;

  // Parameters
  parameter Modelica.Units.SI.Voltage Vin = 14;
  parameter Modelica.Units.SI.Inductance L = 100e-6;
  parameter Modelica.Units.SI.Capacitance C = 100e-6;
  parameter Modelica.Units.SI.Resistance R = 10;
  parameter Modelica.Units.SI.Resistance R_esr = 1.0    "Inductor DCR";

  // RL interfaces
  Modelica.Blocks.Interfaces.RealInput duty_cycle(start = 0.42)
    annotation(Placement(transformation(origin = {-94, 48}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput Vref(start = 5)
    annotation(Placement(transformation(origin = {-94, 0}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealOutput Vout
    annotation(Placement(transformation(origin = {132, -14}, extent = {{-10, -10}, {10, 10}})));

  // Averaged PWM source
  Modelica.Blocks.Math.Product product
    annotation(Placement(transformation(origin = {-30, 40}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Sources.Constant Vin_const(k = Vin)
    annotation(Placement(transformation(origin = {-70, 20}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Sources.SignalVoltage averaged_source
    annotation(Placement(transformation(origin = {-30, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));

  // LC filter + damping
  Modelica.Electrical.Analog.Basic.Inductor inductor(L = L)
    annotation(Placement(transformation(origin = {-4, 0}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Resistor inductor_esr(R = R_esr)
    annotation(Placement(transformation(origin = {22, 0}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Capacitor capacitor(C = C)
    annotation(Placement(transformation(origin = {50, -16}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Resistor resistor(R = R)
    annotation(Placement(transformation(origin = {82, -16}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Electrical.Analog.Basic.Ground ground
    annotation(Placement(transformation(origin = {22, -52}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Sensors.VoltageSensor voltageSensor
    annotation(Placement(transformation(origin = {110, -14}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));

equation
  // Averaged PWM
  connect(duty_cycle, product.u1);
  connect(Vin_const.y, product.u2);
  connect(product.y, averaged_source.v);

  // Power stage: source -> inductor -> esr -> output node
  connect(averaged_source.p, inductor.p);
  connect(inductor.n, inductor_esr.p);
  connect(inductor_esr.n, capacitor.p);
  connect(inductor_esr.n, resistor.p);
  connect(capacitor.n, ground.p);
  connect(resistor.n, ground.p);
  connect(averaged_source.n, ground.p);

  // Sensor
  connect(voltageSensor.p, inductor_esr.n);
  connect(voltageSensor.n, ground.p);
  connect(voltageSensor.v, Vout);

end BuckConverter_Averaged;