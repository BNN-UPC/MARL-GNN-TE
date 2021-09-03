#!/usr/bin/python3

import os

#ENVIRONMENT
networks=['TopologyZoo/Bandcon','TopologyZoo/Ans','TopologyZoo/Aarnet','TopologyZoo/Spiralight','TopologyZoo/Cesnet2001','TopologyZoo/HostwayInternational','TopologyZoo/Rhnet','TopologyZoo/Integra','TopologyZoo/Marnet','TopologyZoo/Darkstrand','TopologyZoo/Packetexchange','TopologyZoo/GtsRomania','TopologyZoo/Noel','TopologyZoo/Restena','TopologyZoo/GtsHungary','TopologyZoo/Aconet','TopologyZoo/Amres','TopologyZoo/Arpanet19719','TopologyZoo/Arpanet19728','TopologyZoo/Garr199905','TopologyZoo/Psinet','TopologyZoo/Nsfnet','TopologyZoo/HurricaneElectric','TopologyZoo/HiberniaUs','TopologyZoo/Garr199904','TopologyZoo/Arn','TopologyZoo/Oxford','TopologyZoo/Abvt','TopologyZoo/Twaren','TopologyZoo/Renater1999','TopologyZoo/Xeex','TopologyZoo/Renater2004','TopologyZoo/Vinaren','TopologyZoo/Ilan','TopologyZoo/VisionNet','TopologyZoo/Sago','TopologyZoo/Ibm','TopologyZoo/Fatman','TopologyZoo/EliBackbone','TopologyZoo/Garr200404','TopologyZoo/KentmanJul2005','TopologyZoo/Shentel','TopologyZoo/WideJpn','TopologyZoo/Cynet','TopologyZoo/Nordu2010','TopologyZoo/Navigata','TopologyZoo/Claranet','TopologyZoo/Biznet','TopologyZoo/BtEurope','TopologyZoo/Arpanet19723','TopologyZoo/Quest','TopologyZoo/Gambia','TopologyZoo/Garr200112','TopologyZoo/Cesnet200304','TopologyZoo/Geant2001','TopologyZoo/KentmanAug2005','TopologyZoo/Peer1','TopologyZoo/Bbnplanet','TopologyZoo/Garr200109','TopologyZoo/Istar','TopologyZoo/Ernet','TopologyZoo/Jgn2Plus','TopologyZoo/Savvis','TopologyZoo/Janetbackbone','TopologyZoo/Agis','TopologyZoo/Uran','TopologyZoo/BtAsiaPac','TopologyZoo/HiberniaUk','TopologyZoo/Sprint','TopologyZoo/Grena','TopologyZoo/Compuserve','TopologyZoo/Atmnet','TopologyZoo/York','TopologyZoo/Goodnet','TopologyZoo/Renater2001']
traffics=['uniform']
routings=['ecmp']
model_dirs=['./checkpoints/training/NSFNet+GEANT2-uniform_paul-ecmp/batch25-lr0.0003-epsilon0.1-gae0.9-clip0.3-gamma0.97-period50-epoch3/size16-iters8-min_max-nnsize64-drop0.25-selu/episode57']
#model_dirs=['./checkpoints/training/NSFNet+GEANT2-uniform_paul-ecmp/batch25-lr0.0003-epsilon0.1-gae0.9-clip0.3-gamma0.97-period50-epoch3/size16-iters8-min_max-nnsize64-drop0.25-tanh/episode35']

#PPOAGENT
batches = [25]
gae_lambdas = [0.9]
lrs = [0.0003]
epsilons = [0.1]
periods=[50]
gammas=[0.95]
clips=[0.2]
epochs=[3]

#ACTOR-CRITIC
link_state_sizes=[16]
message_iterations=[8]
aggregations=['min_max']
nn_sizes = [[64,128]]
dropouts = [0.15]
activations = ['selu']


for activation in activations:
    for dropout in dropouts:
        for nn_size in nn_sizes:
            for aggregation in aggregations:
                for message_iteration in message_iterations:
                    for link_state_size in link_state_sizes:
                        for routing in routings:
                            for model_dir in model_dirs:
                                for traffic in traffics:
                                    for network in networks:
                                        for period in periods:
                                            for gamma in gammas:
                                                for clip in clips:
                                                    for epoch in epochs:
                                                        for batch in batches:
                                                            for gae_lambda in gae_lambdas:
                                                                for lr in lrs:
                                                                    for epsilon in epsilons:
                                                                        activation = 'selu' if 'selu' in model_dir else 'tanh'
                                                                        dropout = 0.15 if 'drop0.15' in model_dir else 0.25
                                                                        cmd = "python ./run.py" \
                                                                            " --gin_bindings='Runner.model_dir = \""+model_dir+"\"'" \
                                                                            " --gin_bindings='Runner.reload_model = True'" \
                                                                            " --gin_bindings='Runner.only_eval = True'" \
                                                                            " --gin_bindings='Runner.save_checkpoints = False'" \
                                                                            " --gin_bindings='Environment.env_type = \""+network+"\"'" \
                                                                            " --gin_bindings='Environment.traffic_profile = \""+traffic+"\"'" \
                                                                            " --gin_bindings='Environment.routing = \""+routing+"\"'" \
                                                                            " --gin_bindings='PPOAgent.eval_period = "+str(period)+"'" \
                                                                            " --gin_bindings='PPOAgent.gamma = "+str(gamma)+"'" \
                                                                            " --gin_bindings='PPOAgent.clip_param = "+str(clip)+"'" \
                                                                            " --gin_bindings='PPOAgent.epochs = "+str(epoch)+"'" \
                                                                            " --gin_bindings='PPOAgent.batch_size = "+str(batch)+"'" \
                                                                            " --gin_bindings='PPOAgent.gae_lambda = "+str(gae_lambda)+"'" \
                                                                            " --gin_bindings='tf.keras.optimizers.Adam.learning_rate = "+str(lr)+"'" \
                                                                            " --gin_bindings='tf.keras.optimizers.Adam.epsilon = "+str(epsilon)+"'" \
                                                                            " --gin_bindings='Actor.link_state_size = "+str(link_state_size)+"'" \
                                                                            " --gin_bindings='Actor.aggregation = \""+aggregation+"\"'" \
                                                                            " --gin_bindings='Actor.first_hidden_layer_size = "+str(nn_size[1])+"'" \
                                                                            " --gin_bindings='Actor.final_hidden_layer_size = "+str(nn_size[0])+"'" \
                                                                            " --gin_bindings='Actor.dropout_rate = "+str(dropout)+"'" \
                                                                            " --gin_bindings='Actor.message_iterations = "+str(message_iteration)+"'" \
                                                                            " --gin_bindings='Actor.activation_fn = \""+activation+"\"'" \
                                                                            " --gin_bindings='Critic.link_state_size = "+str(link_state_size)+"'" \
                                                                            " --gin_bindings='Critic.aggregation = \""+aggregation+"\"'" \
                                                                            " --gin_bindings='Critic.first_hidden_layer_size = "+str(nn_size[1])+"'" \
                                                                            " --gin_bindings='Critic.final_hidden_layer_size = "+str(nn_size[0])+"'" \
                                                                            " --gin_bindings='Critic.dropout_rate = "+str(dropout)+"'" \
                                                                            " --gin_bindings='Critic.message_iterations = "+str(message_iteration)+"'" \
                                                                            " --gin_bindings='Critic.activation_fn = \""+activation+"\"' &"
                                                                        
                                                                        os.system(cmd)     