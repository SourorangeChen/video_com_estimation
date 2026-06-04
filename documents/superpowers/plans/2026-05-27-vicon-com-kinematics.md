# Vicon CoM Kinematics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-frame Vicon CoM displacement, velocity, and xCoM calculations for every trial.

**Architecture:** Keep the calculation as a pure NumPy helper in `video-vicon/code/validate_com_normalized.py`, then add a CSV writer and call it from `main()`. This reuses the existing Vicon parser, manifest loader, and time-axis functions.

**Tech Stack:** Python, NumPy, pytest, Vicon CSV `Model Outputs`.

---

### Task 1: Pure Kinematics Tests

**Files:**
- Modify: `video-vicon/code/tests/test_validate_com_normalized.py`

- [ ] **Step 1: Write failing tests**

Add tests for `compute_com_kinematics` using simple CoM trajectories in meters-equivalent millimeters.

- [ ] **Step 2: Run tests to verify failure**

Run: `D:\Program\Anaconda3\python.exe -m pytest video-vicon/code/tests/test_validate_com_normalized.py -q`

Expected: import failure for `compute_com_kinematics`.

### Task 2: Pure Kinematics Implementation

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`

- [ ] **Step 1: Implement `compute_com_kinematics`**

The function accepts frame numbers, time axis, and three CoM arrays in millimeters. It returns one dict per frame with CoM in meters, displacement, velocity, `omega0`, and xCoM.

- [ ] **Step 2: Run tests to verify pass**

Run: `D:\Program\Anaconda3\python.exe -m pytest video-vicon/code/tests/test_validate_com_normalized.py -q`

Expected: all tests pass.

### Task 3: CSV Output

**Files:**
- Modify: `video-vicon/code/validate_com_normalized.py`

- [ ] **Step 1: Add `write_kinematics_csv` and `compute_trial_kinematics`**

Use the existing parser and manifest time-axis helpers to create per-trial rows.

- [ ] **Step 2: Update `main()`**

Keep existing validation summary output and additionally write `video-vicon/validation/vicon_com_kinematics.csv`.

- [ ] **Step 3: Run the full script**

Run: `D:\Program\Anaconda3\python.exe video-vicon\code\validate_com_normalized.py`

Expected: summary CSV, plots, and kinematics CSV are written.
