python examples/dqn/cartpole-gorila.py --rank 0 --world-size 8 --node-type ps --output models/gorila.pth --memory-rank 1 --actors 2 3 --learners 4 5 6 7 --no-cuda &
python examples/dqn/cartpole-gorila.py --rank 1 --world-size 8 --node-type memory --no-cuda &
python examples/dqn/cartpole-gorila.py --rank 2 --world-size 8 --node-type actor --ps-rank 0 --memory-rank 1 --cuda &
python examples/dqn/cartpole-gorila.py --rank 3 --world-size 8 --node-type actor --ps-rank 0 --memory-rank 1 --cuda &
python examples/dqn/cartpole-gorila.py --rank 4 --world-size 8 --node-type learner --ps-rank 0 --memory-rank 1 --cuda &
python examples/dqn/cartpole-gorila.py --rank 5 --world-size 8 --node-type learner --ps-rank 0 --memory-rank 1 --cuda &
python examples/dqn/cartpole-gorila.py --rank 6 --world-size 8 --node-type learner --ps-rank 0 --memory-rank 1 --cuda &
python examples/dqn/cartpole-gorila.py --rank 7 --world-size 8 --node-type learner --ps-rank 0 --memory-rank 1 --cuda &