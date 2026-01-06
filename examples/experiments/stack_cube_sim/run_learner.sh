export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python ../../train_rlpd.py "$@" \
    --exp_name=stack_cube_sim \
    --checkpoint_path=run7 \
    --demo_path=/home/wayne/hil-serl-sim2/examples/experiments/stack_cube_sim/demo_data/stack_cube_sim_20_demos_2025-12-17_16-32-47.pkl \
    --learner \