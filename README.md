# Self Playing RNA Inverse Folding

Self Play (SP) is a method in Reinforcement Learning where an agent learns by playing against itself until state-action or state-value evaluation 
functions converge. The SP method has recorded state of the art results in playing different computer games including Chess, Go and Othello. 
In this paper, we propose a SP based agent dubbed Self Player RNA (SPRNA) applied to Ribonucleic acid (RNA) Inverse Folding where a sequence is 
designed to match a given target structure. We also present an encoding scheme for both the known and unknown RNA Inverse Folding states and the 
corresponding Gym Environment. We show that the optimal policies in RNA Inverse Folding can be learned by just performing a one-step look-ahead 
state-value evaluation using a deep value network. SPRNA learns to design RNA sequences recording very competitive results across different 
RNA sequence design  benchmark datasets
