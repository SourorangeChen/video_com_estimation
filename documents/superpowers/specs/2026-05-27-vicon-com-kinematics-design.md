# Vicon CoM Kinematics Design

## Goal

Compute per-trial Vicon center-of-mass kinematics from the existing Vicon `Model Outputs/:CentreOfMass` data: displacement, velocity, and extrapolated center of mass (xCoM).

## Data Source

Use the existing Vicon CSV parser in `video-vicon/code/validate_com_normalized.py`. The parser reads `CentreOfMass` coordinates in millimeters from each trial CSV. The manifest supplies each trial's first Vicon frame and trajectory rate.

## Calculations

All kinematic calculations convert Vicon CoM from millimeters to meters.

- Displacement vector: `r(t_i) - r(t_{i-1})`; the first frame is zero
- Displacement magnitude: `sqrt(dx^2 + dy^2 + dz^2)`
- Velocity vector: numerical derivative `dr/dt` with `np.gradient` over the Vicon time axis
- Velocity magnitude: `sqrt(vx^2 + vy^2 + vz^2)`
- Pendulum length: `l(t) = com_z(t)` in meters for each frame
- Natural frequency: `omega0(t) = sqrt(g / l(t))`, using `g = 9.81 m/s^2`
- xCoM: `CoM(t) + velocity(t) / omega0(t)`

If any frame has non-positive `com_z`, the calculation raises a `ValueError` because `omega0` would be invalid.

## Output

Write `video-vicon/validation/vicon_com_kinematics.csv`, one row per Vicon frame:

`trial, frame, time_s, com_x_m, com_y_m, com_z_m, displacement_x_m, displacement_y_m, displacement_z_m, displacement_m, velocity_x_m_s, velocity_y_m_s, velocity_z_m_s, velocity_m_s, omega0_rad_s, xcom_x_m, xcom_y_m, xcom_z_m`

## Testing

Add focused unit tests for the pure kinematics function before implementation:

- displacement is relative to the previous frame
- velocity uses the supplied time axis
- `omega0` uses each frame's CoM height
- xCoM equals `CoM + velocity / omega0`
- non-positive height raises a clear error
