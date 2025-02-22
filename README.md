# **Awesome ICLR 2025 Graph Paper Collection**

This repo contains a comprehensive compilation of **graph and/or GNN** papers that were accepted at the [The Thirteenth International Conference on Learning Representations 2025](https://iclr.cc/). Graph or Geometric machine learning possesses an indispensable role within the domain of machine learning research, providing invaluable insights, methodologies, and solutions to a diverse array of challenges and problems. 

**Short Overview**: ICLR 2025 showcased exciting advancements in graph research, focusing on Graph Neural Networks (GNNs) and their theoretical foundations. Key themes included enhancing expressivity and generalization, with many papers building on Weisfeiler-Lehman methods and spectral analyses. Hypergraph techniques and innovative message-passing approaches were explored to capture higher-order relationships, while diffusion and flow matching models showed promise for graph generation and transformation tasks. Equivariant architectures gained attention for ensuring symmetry and robustness in complex settings. Robustness strategies, including out-of-distribution detection and anomaly identification, were prominently featured. The integration of large language models with graph reasoning expanded multi-modal and knowledge-driven applications, with interdisciplinary impacts across domains like PDEs, molecular modeling, and scene graph generation. Overall, the conference highlighted a balanced pursuit of deep theoretical insights alongside scalable, real-world solutions in graph research.


**Have a look and throw me a review (and, a star ⭐, maybe!)** Thanks!


---



## **All Topics:** 

<details open>
  <summary><b>View Topic list!</b></summary>

- [GNN Theories](#theories)
  - [Weisfeiler Leman](#Weisfeiler-Leman )
  - [Hypergraph](#Hypergraph)
  - [Expressivity](#Expressivity)
  - [Generalization](#Generalization)
  - [Equivariant Graph Neural Networks](#Equivariant)
  - [Out-of-Distribution](#OOD)
  - [Diffusion](#Diffusion)
  - [Graph Matching](#GraphMatching)
  - [Flow Matching](#FlowMatching)
  - [Contrastive Learning](#ContrastiveLearning)
  - [Clustering](#Clustering)
  - [Message Passing Neural Networks](#MPNN)
  - [Transformers](#GraphTransformers)
  - [Optimal Transport](#OptimalTransport)
  - [Graph Generation](#ggen)
  - [Unsupervised Learning](#UL)
  - [Meta-learning](#GraphMeta-learning)
  - [Disentanglement](#Disentanglement)
  - [Others](#GNNT-others)
- [GNNs for PDE/ODE/Physics](#PDE)
- [Graph and Large Language Models/Agents](#LLM)
- [Knowledge Graph and Knowledge Graph Embeddings](#KG)
- [GNN Applications](#apps)
- [Spatial and/or Temporal GNNs](#SpatialTemporalGNNs)
- [Explainable AI](#xai)
- [Reinforcement Learning](#rl)
- [Graphs, Molecules and Biology](#molecular)
- [GFlowNets](#GFlowNets)
- [Causal Discovery and Graphs](#Causal)
- [Federated Learning, Privacy, Decentralization](#FL)
- [Scene Graphs](#SceneGraphs)
- [Graphs, GNNs and Efficiency](#Efficiency)
- [Others](#Others)
- [More Possible Works](#more)
</details>


<a name="theories" />

## GNN Theories 

<a name="Weisfeiler-Leman " />

#### Weisfeiler Leman 
- [Generalizing Weisfeiler-Lehman Kernels to Subgraphs](https://openreview.net/pdf?id=HZgZrtIreg)


#### Hypergraph
- [DistillHGNN: A Knowledge Distillation Approach for High-Speed Hypergraph Neural Networks](https://openreview.net/pdf?id=vzrs42hgb0)
- [Beyond Mere Token Analysis: A Hypergraph Metric Space Framework for Defending Against Socially Engineered LLM Attacks](https://openreview.net/pdf?id=rnJxelIZrq)
- [HyperPLR: Hypergraph Generation through Projection, Learning, and Reconstruction](https://openreview.net/pdf?id=TYnne6Pa35)
- [The Ramanujan Library - Automated Discovery on the Hypergraph of Integer Relations](https://openreview.net/pdf?id=EyaH1wzmao)
- [Training-Free Message Passing for Learning on Hypergraphs](https://openreview.net/pdf?id=4AuyYxt7A2)
- [Beyond Graphs: Can Large Language Models Comprehend Hypergraphs?](https://openreview.net/pdf?id=28qOQwjuma)

#### Expressivity
- [Homomorphism Expressivity of Spectral Invariant Graph Neural Networks](https://openreview.net/pdf?id=rdv6yeMFpn)
- [Generalization, Expressivity, and Universality of Graph Neural Networks on Attributed Graphs](https://openreview.net/pdf?id=qKgd7RaAem)
- [Is uniform expressivity too restrictive? Towards efficient expressivity of GNNs](https://openreview.net/pdf?id=lsvGqR6OTf)
- [Topological Blindspots: Understanding and Extending Topological Deep Learning Through the Lens of Expressivity](https://openreview.net/pdf?id=EzjsoomYEb)
- [Towards Bridging Generalization and Expressivity of Graph Neural Networks](https://openreview.net/pdf?id=BOQpRtI4F5)
- [On the Expressive Power of Tree-Structured Probabilistic Circuits](https://openreview.net/pdf?id=suYAAOI5bd)
- [On the Expressive Power of Sparse Geometric MPNNs](https://openreview.net/pdf?id=NY7aEek0mi)

#### Generalization
- [N-ForGOT: Towards Not-forgetting and Generalization of Open Temporal Graph Learning](https://openreview.net/pdf?id=rLlDt2FQvz)
- [Generalization, Expressivity, and Universality of Graph Neural Networks on Attributed Graphs](https://openreview.net/pdf?id=qKgd7RaAem)
- [Towards Generalization Bounds of GCNs for Adversarially Robust Node Classification](https://openreview.net/pdf?id=cp3aW7C5tD)
- [Subgraph Federated Learning for Local Generalization](https://openreview.net/pdf?id=cH65nS5sOz)
- [Generalization and Distributed Learning of GFlowNets](https://openreview.net/pdf?id=PJNhZoCjLh)
- [Towards Bridging Generalization and Expressivity of Graph Neural Networks](https://openreview.net/pdf?id=BOQpRtI4F5)


<a name="Equivariant" />

#### Equivariant Graph Neural Networks
- [Equivariant Denoisers Cannot Copy Graphs: Align Your Graph Diffusion Models](https://openreview.net/pdf?id=onIro14tHv)
- [E(3)-equivariant models cannot learn chirality: Field-based molecular generation](https://openreview.net/pdf?id=mXHTifc1Fn)
- [Improving Equivariant Networks with Probabilistic Symmetry Breaking](https://openreview.net/pdf?id=ZE6lrLvATd)
- [Learning Equivariant Non-Local Electron Density Functionals](https://openreview.net/pdf?id=FhBT596F1X)
- [E(n) Equivariant Topological Neural Networks](https://openreview.net/pdf?id=Ax3uliEBVR)
- [GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks](https://openreview.net/pdf?id=5wxCQDtbMo)
- [Graph Neural Networks for Edge Signals: Orientation Equivariance and Invariance](https://openreview.net/pdf?id=XWBE90OYlH)
- [REVISITING MULTI-PERMUTATION EQUIVARIANCE THROUGH THE LENS OF IRREDUCIBLE REPRESENTATIONS](https://openreview.net/pdf?id=4v4nmYWzBa)

<a name="OOD" />

#### Out-of-Distribution
- [GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation](https://openreview.net/pdf?id=y5einmJ0Yx)
- [A new framework for evaluating model out-of-distribution generalisation for the biochemical domain](https://openreview.net/pdf?id=qFZnAC4GHR)
- [Spreading Out-of-Distribution Detection on Graphs](https://openreview.net/pdf?id=p1TBYyqy8v)
- [Unifying Unsupervised Graph-Level Anomaly Detection and Out-of-Distribution Detection: A Benchmark](https://openreview.net/pdf?id=g90RNzs8wX)
- [Decoupled Graph Energy-based Model for Node Out-of-Distribution Detection on Heterophilic Graphs](https://openreview.net/pdf?id=NuVBI4wPMm)
- [BrainOOD: Out-of-distribution Generalizable Brain Network Analysis](https://openreview.net/pdf?id=3xqqYOKILp)

<a name="GraphMatching" />

#### Graph Matching
- [Learning Partial Graph Matching via Optimal Partial Transport](https://openreview.net/pdf?id=uDXFOurrHM)
- [Charting the Design Space of Neural Graph Representations for Subgraph Matching](https://openreview.net/pdf?id=5pd78GmXC6)
- [PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching](https://openreview.net/pdf?id=27Qk18IZum)

<a name="FlowMatching" />

#### Flow Matching
- [AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly](https://openreview.net/pdf?id=jckKNzYYA6)
- [Stiefel Flow Matching for Moment-Constrained Structure Elucidation](https://openreview.net/pdf?id=84WmbzikPP)

<a name="ContrastiveLearning" />

#### Contrastive Learning
- [Aligning Visual Contrastive learning models via Preference Optimization](https://openreview.net/pdf?id=wgRQ2WAORJ)
- [CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening](https://openreview.net/pdf?id=fv9XU7CyN2)
- [$\\mathbb{X}$-Sample Contrastive Loss: Improving Contrastive Learning with Sample Similarity Graphs](https://openreview.net/pdf?id=c1Ng0f8ivn)

### Diffusion
- [Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition](https://openreview.net/pdf?id=zZ6TT254Np)
- [SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models](https://openreview.net/pdf?id=xnssGv9rpW)
- [Advancing Graph Generation through Beta Diffusion](https://openreview.net/pdf?id=x1An5a3U9I)
- [Self-Supervised Diffusion MRI Denoising via Iterative and Stable Refinement](https://openreview.net/pdf?id=wxPnuFp8fZ)
- [Diffusion On Syntax Trees For Program Synthesis](https://openreview.net/pdf?id=wN3KaUXA5X)
- [Learning Distributions of Complex Fluid Simulations with Diffusion Graph Networks](https://openreview.net/pdf?id=uKZdlihDDn)
- [Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them](https://openreview.net/pdf?id=tZdqL5FH7w)
- [Discrete Diffusion Schrodinger Bridge Matching for Graph Transformation](https://openreview.net/pdf?id=tQyh0gnfqW)
- [U-Nets as Belief Propagation: Efficient Classification, Denoising, and Diffusion in Generative Hierarchical Models](https://openreview.net/pdf?id=sy1lbQxj9J)
- [Bundle Neural Network for message diffusion on graphs](https://openreview.net/pdf?id=scI9307PLG)
- [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](https://openreview.net/pdf?id=p66a00KLWN)
- [Equivariant Denoisers Cannot Copy Graphs: Align Your Graph Diffusion Models](https://openreview.net/pdf?id=onIro14tHv)
- [Topological Zigzag Spaghetti for Diffusion-based Generation and Prediction on Graphs](https://openreview.net/pdf?id=mYgoNEsUDi)
- [LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation](https://openreview.net/pdf?id=kam84eEmub)
- [Instant Policy: In-Context Imitation Learning via Graph Diffusion](https://openreview.net/pdf?id=je3GZissZc)
- [Simple and Controllable Uniform Discrete Diffusion Language Models](https://openreview.net/pdf?id=i5MrJ6g5G1)
- [Retrieval Augmented Diffusion Model for Structure-informed Antibody Design and Optimization](https://openreview.net/pdf?id=a6U41REOa5)
- [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://openreview.net/pdf?id=XsgHl54yO7)
- [Self-Supervised Diffusion Processes for Electron-Aware Molecular Representation Learning](https://openreview.net/pdf?id=UQ0RqfhgCk)
- [InverseBench: Benchmarking Plug-and-Play Diffusion Models for Scientific Inverse Problems](https://openreview.net/pdf?id=U3PBITXNG6)
- [Relation-Aware Diffusion for Heterogeneous Graphs with Partially Observed Features](https://openreview.net/pdf?id=TPYwwqF0bv)
- [Fast Direct: Query-Efficient  Online Black-box Guidance  for Diffusion-model Target Generation](https://openreview.net/pdf?id=OmpTdjl7RV)
- [Erasing Concept Combination from Text-to-Image Diffusion Model](https://openreview.net/pdf?id=OBjF5I4PWg)
- [TANGO: Co-Speech Gesture Video Reenactment with Hierarchical Audio Motion Embedding and Diffusion Interpolation](https://openreview.net/pdf?id=LbEWwJOufy)
- [Bias Mitigation in Graph Diffusion Models](https://openreview.net/pdf?id=CSj72Rr2PB)
- [Periodic Materials Generation using Text-Guided Joint Diffusion Model](https://openreview.net/pdf?id=AkBrb7yQ0G)
- [GRASP: Generating  Graphs  via Spectral Diffusion](https://openreview.net/pdf?id=AAXBfJNHDt)
- [An Efficient Framework for Crediting Data Contributors of Diffusion Models](https://openreview.net/pdf?id=9EqQC2ct4H)
- [DPLM-2: A Multimodal Diffusion Protein Language Model](https://openreview.net/pdf?id=5z9GjHgerY)
- [Chemistry-Inspired Diffusion with Non-Differentiable Guidance](https://openreview.net/pdf?id=4dAgG8ma3B)

#### Clustering
- [On the Price of Differential Privacy for Hierarchical Clustering](https://openreview.net/pdf?id=yLhJYvkKA0)
- [Simple yet Effective Incomplete Multi-view Clustering: Similarity-level Imputation and Intra-view Hybrid-group Prototype Construction](https://openreview.net/pdf?id=KijslFbfOL)
- [Demystifying Online Clustering of Bandits: Enhanced Exploration Under Stochastic and Smoothed Adversarial Contexts](https://openreview.net/pdf?id=421D67DY3i)
- [Coreset Spectral Clustering](https://openreview.net/pdf?id=1qgZXeMTTU)

<a name="MPNN" />

#### Message Passing Neural Networks
- [On the Expressive Power of Sparse Geometric MPNNs](https://openreview.net/pdf?id=NY7aEek0mi)
- [PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems](https://openreview.net/pdf?id=fU8H4lzkIm)
- [Beyond Circuit Connections: A Non-Message Passing Graph Transformer Approach for Quantum Error Mitigation](https://openreview.net/pdf?id=XnVttczoAV)
- [Training-Free Message Passing for Learning on Hypergraphs](https://openreview.net/pdf?id=4AuyYxt7A2)


<a name="GraphTransformers" />

#### Transformers
- [Graph Transformers Dream of Electric Flow](https://openreview.net/pdf?id=rWQDzq3O5c)
- [Learning Graph Quantized Tokenizers for Transformers](https://openreview.net/pdf?id=oYSsbY3G4o)
- [RNNs are not Transformers (Yet):  The Key Bottleneck on In-Context Retrieval](https://openreview.net/pdf?id=h3wbI8Uk1Z)
- [Learning Randomized Algorithms with Transformers](https://openreview.net/pdf?id=UV5p3JZMjC)
- [Transformers Struggle to Learn to Search Without In-context Exploration](https://openreview.net/pdf?id=9cQB1Hwrtw)
- [Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition](https://openreview.net/pdf?id=41HlN8XYM5)
- [Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data?](https://openreview.net/pdf?id=1Xg4JPPxJ0)
- [A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language](https://openreview.net/pdf?id=0pLCDJVVRD)


<a name="OptimalTransport" />

#### Optimal Transport
- [Linear Spherical Sliced Optimal Transport: A Fast Metric for Comparing Spherical Data](https://openreview.net/pdf?id=fgUFZAxywx)
- [Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport](https://openreview.net/pdf?id=VGURexnlUL)


<a name="ggen" />

#### Graph Generation
- [Advancing Graph Generation through Beta Diffusion](https://openreview.net/pdf?id=x1An5a3U9I)
- [MAGE: Model-Level Graph Neural Networks Explanations via Motif-based Graph Generation](https://openreview.net/pdf?id=vue9P1Ypk6)
- [Lift Your Molecules: Molecular Graph Generation in Latent Euclidean Space](https://openreview.net/pdf?id=uNomADvF3s)
- [Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](https://openreview.net/pdf?id=tpD1rs25Uu)
- [Temporal Heterogeneous Graph Generation with Privacy, Utility, and Efficiency](https://openreview.net/pdf?id=tj5xJInWty)
- [LayerDAG: A Layerwise Autoregressive Diffusion Model for Directed Acyclic Graph Generation](https://openreview.net/pdf?id=kam84eEmub)
- [HyperPLR: Hypergraph Generation through Projection, Learning, and Reconstruction](https://openreview.net/pdf?id=TYnne6Pa35)


<a name="UL" />

#### Unsupervised Learning
- [Learning to Explore and Exploit with GNNs for Unsupervised Combinatorial Optimization](https://openreview.net/pdf?id=vaJ4FObpXN)
- [Unifying Unsupervised Graph-Level Anomaly Detection and Out-of-Distribution Detection: A Benchmark](https://openreview.net/pdf?id=g90RNzs8wX)
- [SPDIM: Source-Free Unsupervised Conditional and Label Shift Adaptation in EEG](https://openreview.net/pdf?id=CoQw1dXtGb)
- [Unsupervised Multiple Kernel Learning for Graphs via Ordinality Preservation](https://openreview.net/pdf?id=6nb2J90XJD)
- [BrainUICL: An Unsupervised Individual Continual Learning Framework for EEG Applications](https://openreview.net/pdf?id=6jjAYmppGQ)


<a name="GraphMeta-learning" />

#### Graph Meta-learning
- [A Meta-Learning Approach to Bayesian Causal Discovery](https://openreview.net/pdf?id=eeJz7eDWKO)
- [MetaDesigner: Advancing Artistic Typography through AI-Driven, User-Centric, and Multilingual WordArt Synthesis](https://openreview.net/pdf?id=Mv3GAYJGcW)
- [Systems with Switching Causal Relations: A Meta-Causal Perspective](https://openreview.net/pdf?id=J9VogDTa1W)

#### Disentanglement
- [Estimation of single-cell and tissue perturbation effect in spatial transcriptomics via Spatial Causal Disentanglement](https://openreview.net/pdf?id=Tqdsruwyac)
- [CFD: Learning Generalized Molecular Representation via Concept-Enhanced  Feedback Disentanglement](https://openreview.net/pdf?id=CsOIYMOZaV)
- [An Information Criterion for Controlled Disentanglement of Multimodal Data](https://openreview.net/pdf?id=3n4RY25UWP)

<a name="PDE" />

## GNNs for PDE/ODE/Physics
- [SINGER: Stochastic Network Graph Evolving Operator for High Dimensional PDEs](https://openreview.net/pdf?id=wVADj7yKee)
- [PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems](https://openreview.net/pdf?id=fU8H4lzkIm)
- [Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics](https://openreview.net/pdf?id=VsxbWTDHjh)
- [Revolutionizing EMCCD Denoising through a Novel Physics-Based Learning Framework for Noise Modeling](https://openreview.net/pdf?id=vmulbBDCan)
- [PIORF: Physics-Informed Ollivier-Ricci Flow for LongRange Interactions in Mesh Graph Neural Networks](https://openreview.net/pdf?id=qkBBHixPow)
- [MeshMask: Physics-Based Simulations with Masked Graph Neural Networks](https://openreview.net/pdf?id=bFHR8hNk4I)
- [Predicting the Energy Landscape of Stochastic Dynamical System via  Physics-informed Self-supervised Learning](https://openreview.net/pdf?id=PxRATSTDlS)


<a name="LLM" />

## Graph and Large Language Models/Agents
- [LICO: Large Language Models for In-Context Molecular Optimization](https://openreview.net/pdf?id=yu1vqQqKkx)
- [Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](https://openreview.net/pdf?id=xoXn62FzD0)
- [Improving Large Language Model based  Multi-Agent Framework through Dynamic Workflow Updating](https://openreview.net/pdf?id=sLKDbuyq99)
- [OCEAN: Offline Chain-of-thought Evaluation and Alignment in Large Language Models](https://openreview.net/pdf?id=rlgplAuN2p)
- [PaLD: Detection of Text Partially Written by Large Language Models](https://openreview.net/pdf?id=rWjZWHYPcz)
- [Reasoning of Large Language Models over Knowledge Graphs with Super-Relations](https://openreview.net/pdf?id=rTCJ29pkuA)
- [Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning](https://openreview.net/pdf?id=rQ7fz9NO7f)
- [Knowledge Graph Finetuning Enhances Knowledge Manipulation in Large Language Models](https://openreview.net/pdf?id=oMFOKjwaRS)
- [Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation](https://openreview.net/pdf?id=oFBu7qaZpS)
- [Can Large Language Models Understand Symbolic Graphics Programs?](https://openreview.net/pdf?id=Yk87CwhBDx)
- [GraphArena: Evaluating and Exploring Large Language Models on Graph Computation](https://openreview.net/pdf?id=Y1r9yCMzeA)
- [Decision Information Meets Large Language Models: The Future of Explainable Operations Research](https://openreview.net/pdf?id=W2dR6rypBQ)
- [Logical Consistency of Large Language Models in Fact-Checking](https://openreview.net/pdf?id=SimlDuN0YT)
- [Scaling Large Language Model-based Multi-Agent Collaboration](https://openreview.net/pdf?id=K3n5jPkrU6)
- [Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation](https://openreview.net/pdf?id=JvkuZZ04O7)
- [RetroInText: A Multimodal Large Language Model Enhanced Framework for Retrosynthetic Planning via In-Context Representation Learning](https://openreview.net/pdf?id=J6e4hurEKd)
- [Quantitative Certification of Bias in Large Language Models](https://openreview.net/pdf?id=HQHnhVQznF)
- [How Do Large Language Models Understand Graph Patterns? A Benchmark for Graph Pattern Comprehension](https://openreview.net/pdf?id=CkKEuLmRnr)
- [OSDA Agent: Leveraging Large Language Models for De Novo Design of Organic Structure Directing Agents](https://openreview.net/pdf?id=9YNyiCJE3k)
- [Beyond Graphs: Can Large Language Models Comprehend Hypergraphs?](https://openreview.net/pdf?id=28qOQwjuma)
- [LLM-based Typed Hyperresolution for Commonsense Reasoning with Knowledge Bases](https://openreview.net/pdf?id=wNobG8bV5Q)
- [Beyond Mere Token Analysis: A Hypergraph Metric Space Framework for Defending Against Socially Engineered LLM Attacks](https://openreview.net/pdf?id=rnJxelIZrq)
- [Encryption-Friendly LLM Architecture](https://openreview.net/pdf?id=pbre0HKsfE)
- [Enhancing Graph Of Thought: Enhancing Prompts with LLM Rationales and Dynamic Temperature Control](https://openreview.net/pdf?id=l32IrJtpOP)
- [GraphRouter: A Graph-based Router for LLM Selections](https://openreview.net/pdf?id=eU39PDsZtT)
- [Mask-DPO: Generalizable Fine-grained Factuality Alignment of LLMs](https://openreview.net/pdf?id=d2H1oTNITn)
- [Searching for Optimal Solutions with LLMs via Bayesian Optimization](https://openreview.net/pdf?id=aVfDrl7xDV)
- [Permute-and-Flip: An optimally stable and watermarkable decoder for LLMs](https://openreview.net/pdf?id=YyVVicZ32M)
- [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://openreview.net/pdf?id=LkzuPorQ5L)
- [HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment](https://openreview.net/pdf?id=Cs6MrbFuMq)
- [$R^2$-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning](https://openreview.net/pdf?id=CkgKSqZbuC)
- [Human-like Episodic Memory for Infinite Context LLMs](https://openreview.net/pdf?id=BI2int5SAC)
- [GraphEval: A Lightweight Graph-Based LLM Framework for Idea Evaluation](https://openreview.net/pdf?id=5RUM1aIdok)


<a name="KG" />

## Knowledge Graph and Knowledge Graph Embeddings
- [Knowledge Graph Based Agent For Complex, Knowledge-Intensive QA in Medicine](https://openreview.net/pdf?id=tnB94WQGrn)
- [Reasoning of Large Language Models over Knowledge Graphs with Super-Relations](https://openreview.net/pdf?id=rTCJ29pkuA)
- [Knowledge Graph Finetuning Enhances Knowledge Manipulation in Large Language Models](https://openreview.net/pdf?id=oMFOKjwaRS)
- [Towards Synergistic Path-based Explanations for Knowledge Graph Completion: Exploration and Evaluation](https://openreview.net/pdf?id=WQvkqarwXi)
- [INFER: A Neural-symbolic Model For Extrapolation Reasoning on Temporal Knowledge Graph](https://openreview.net/pdf?id=ExHUtB2vnz)
- [Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval](https://openreview.net/pdf?id=8fLgt7PQza)



<a name="SpatialTemporalGNNs" />

## Spatial and/or Temporal GNNs
- [Learning Successor Features with Distributed Hebbian Temporal Memory](https://openreview.net/pdf?id=wYJII5BRYU)
- [Temporal Heterogeneous Graph Generation with Privacy, Utility, and Efficiency](https://openreview.net/pdf?id=tj5xJInWty)
- [N-ForGOT: Towards Not-forgetting and Generalization of Open Temporal Graph Learning](https://openreview.net/pdf?id=rLlDt2FQvz)
- [PhyMPGN: Physics-encoded Message Passing Graph Network for spatiotemporal PDE systems](https://openreview.net/pdf?id=fU8H4lzkIm)
- [LASER: A Neuro-Symbolic Framework for Learning Spatio-Temporal Scene Graphs with Weak Supervision](https://openreview.net/pdf?id=HEXtydywnE)
- [Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting](https://openreview.net/pdf?id=FRzCIlkM7I)
- [INFER: A Neural-symbolic Model For Extrapolation Reasoning on Temporal Knowledge Graph](https://openreview.net/pdf?id=ExHUtB2vnz)
- [TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics](https://openreview.net/pdf?id=8e2LirwiJT)




<a name="apps" />

## GNN Applications
- [On Designing General and Expressive Quantum Graph Neural Networks with Applications to MILP Instance Representation](https://openreview.net/pdf?id=IQi8JOqLuv)
- [BrainUICL: An Unsupervised Individual Continual Learning Framework for EEG Applications](https://openreview.net/pdf?id=6jjAYmppGQ)
<a name="xai" />

## Explainable AI
- [Reconsidering Faithfulness in Regular, Self-Explainable and Domain Invariant GNNs](https://openreview.net/pdf?id=kiOxNsrpQy)
- [Provably Robust Explainable Graph Neural Networks against Graph Perturbation Attacks](https://openreview.net/pdf?id=iFK0xoceR0)
- [Decision Information Meets Large Language Models: The Future of Explainable Operations Research](https://openreview.net/pdf?id=W2dR6rypBQ)
- [XAIguiFormer: explainable artificial intelligence guided transformer for brain disorder identification](https://openreview.net/pdf?id=AD5yx2xq8R)
- [XAIguiFormer: explainable artificial intelligence guided transformer for brain disorder identification](https://openreview.net/pdf?id=AD5yx2xq8R)

<a name="rl" />

## Reinforcement Learning
- [Grammar Reinforcement Learning: path and cycle counting in graphs with a Context-Free Grammar and Transformer approach](https://openreview.net/pdf?id=yEox25xAED)
- [Learning Splitting Heuristics in Divide-and-Conquer SAT Solvers with Reinforcement Learning](https://openreview.net/pdf?id=uUsL07BsMA)
- [Data Center Cooling System Optimization Using Offline Reinforcement Learning](https://openreview.net/pdf?id=W8xukd70cU)
- [Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning](https://openreview.net/pdf?id=CL3U0GxFRD)
- [Sequential Stochastic Combinatorial Optimization Using Hierarchal Reinforcement Learning](https://openreview.net/pdf?id=AloCXPpq54)
- [Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling](https://openreview.net/pdf?id=4PlbIfmX9o)
- [Syntactic and Semantic Control of Large Language Models via Sequential Monte Carlo](https://openreview.net/pdf?id=xoXn62FzD0)
- [CausalRivers - Scaling up benchmarking of causal discovery for real-world time-series](https://openreview.net/pdf?id=wmV4cIbgl6)
- [OCEAN: Offline Chain-of-thought Evaluation and Alignment in Large Language Models](https://openreview.net/pdf?id=rlgplAuN2p)
- [N-ForGOT: Towards Not-forgetting and Generalization of Open Temporal Graph Learning](https://openreview.net/pdf?id=rLlDt2FQvz)
- [Interleaved Scene Graph for Interleaved Text-and-Image Generation Assessment](https://openreview.net/pdf?id=rDLgnYLM5b)
- [Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents](https://openreview.net/pdf?id=kxnoqaisCT)
- [Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs](https://openreview.net/pdf?id=i8vPRlsrYu)
- [FACTS: A Factored State-Space Framework for World Modelling](https://openreview.net/pdf?id=dmCGjPFVhF)
- [Searching for Optimal Solutions with LLMs via Bayesian Optimization](https://openreview.net/pdf?id=aVfDrl7xDV)
- [Improving Equivariant Networks with Probabilistic Symmetry Breaking](https://openreview.net/pdf?id=ZE6lrLvATd)
- [ADAM: An Embodied Causal Agent in Open-World Environments](https://openreview.net/pdf?id=Ouu3HnIVBc)
- [Geometry-aware RL for Manipulation of Varying Shapes and Deformable Objects](https://openreview.net/pdf?id=7BLXhmWvwF)
- [Multi-Label Node Classification with Label Influence Propagation](https://openreview.net/pdf?id=3X3LuwzZrl)


<a name="molecular" />

## Graphs and Molecules
- [LICO: Large Language Models for In-Context Molecular Optimization](https://openreview.net/pdf?id=yu1vqQqKkx)
- [DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-training](https://openreview.net/pdf?id=ym7pr83XQr)
- [MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy Spectra](https://openreview.net/pdf?id=xJDxVDG3x2)
- [Lift Your Molecules: Molecular Graph Generation in Latent Euclidean Space](https://openreview.net/pdf?id=uNomADvF3s)
- [Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems](https://openreview.net/pdf?id=twEvvkQqPS)
- [Multimodal Large Language Models for Inverse Molecular Design with Retrosynthetic Planning](https://openreview.net/pdf?id=rQ7fz9NO7f)
- [InversionGNN: A Dual Path Network for Multi-Property Molecular Optimization](https://openreview.net/pdf?id=nYPuSzGE3X)
- [E(3)-equivariant models cannot learn chirality: Field-based molecular generation](https://openreview.net/pdf?id=mXHTifc1Fn)
- [AssembleFlow: Rigid Flow Matching with Inertial Frames for Molecular Assembly](https://openreview.net/pdf?id=jckKNzYYA6)
- [CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening](https://openreview.net/pdf?id=fv9XU7CyN2)
- [REBIND: Enhancing Ground-state Molecular Conformation Prediction via Force-Based Graph Rewiring](https://openreview.net/pdf?id=WNIEr5kydF)
- [Self-Supervised Diffusion Processes for Electron-Aware Molecular Representation Learning](https://openreview.net/pdf?id=UQ0RqfhgCk)
- [CFD: Learning Generalized Molecular Representation via Concept-Enhanced  Feedback Disentanglement](https://openreview.net/pdf?id=CsOIYMOZaV)
- [Learning Molecular Representation in a Cell](https://openreview.net/pdf?id=BbZy8nI1si)
- [MADGEN - Mass-Spec attends to De Novo Molecular generation](https://openreview.net/pdf?id=78tc3EiUrN)
- [Curriculum-aware Training for Discriminating Molecular Property Prediction Models](https://openreview.net/pdf?id=6DHIkLv5i3)
- [Iterative Substructure Extraction for Molecular Relational Learning with Interactive Graph Information Bottleneck](https://openreview.net/pdf?id=3kiZ5S5WkY)
- [DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based 3D Molecular Pre-training](https://openreview.net/pdf?id=ym7pr83XQr)
- [SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints](https://openreview.net/pdf?id=uvHmnahyp1)
- [Lift Your Molecules: Molecular Graph Generation in Latent Euclidean Space](https://openreview.net/pdf?id=uNomADvF3s)
- [NExT-Mol: 3D Diffusion Meets 1D Language Modeling for 3D Molecule Generation](https://openreview.net/pdf?id=p66a00KLWN)
- [Atomas: Hierarchical Adaptive Alignment on Molecule-Text for Unified Molecule Understanding and Generation](https://openreview.net/pdf?id=mun3bGqdDM)
- [CBGBench: Fill in the Blank of Protein-Molecule Complex Binding Graph](https://openreview.net/pdf?id=mOpNrrV2zH)
- [Fragment and Geometry Aware Tokenization of Molecules for Structure-Based Drug Design Using Language Models](https://openreview.net/pdf?id=mMhZS7qt0U)
- [Leveraging Discrete Structural Information for Molecule-Text Modeling](https://openreview.net/pdf?id=eGqQyTAbXC)
- [Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport](https://openreview.net/pdf?id=VGURexnlUL)
- [A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](https://openreview.net/pdf?id=OIvg3MqWX2)
- [Procedural Synthesis of Synthesizable Molecules](https://openreview.net/pdf?id=OGfyzExd69)
- [MAGNet: Motif-Agnostic Generation of Molecules from Scaffolds](https://openreview.net/pdf?id=5FXKgOxmb2)

<a name="GFlowNets" />

## **GFlowNets**
- [Generalization and Distributed Learning of GFlowNets](https://openreview.net/pdf?id=PJNhZoCjLh)
- [When do GFlowNets learn the right distribution?](https://openreview.net/pdf?id=9GsgCUJtic)


<a name="Causal" />

## Causal Discovery and Graphs
- [When Selection meets Intervention: Additional Complexities in Causal Discovery](https://openreview.net/pdf?id=xByvdb3DCm)
- [Which Tasks Should Be Compressed Together? A Causal Discovery Approach for Efficient Multi-Task Representation Compression](https://openreview.net/pdf?id=x33vSZUg0A)
- [CausalRivers - Scaling up benchmarking of causal discovery for real-world time-series](https://openreview.net/pdf?id=wmV4cIbgl6)
- [Identifiable Exchangeable Mechanisms for Causal Structure and Representation Learning](https://openreview.net/pdf?id=k03mB41vyM)
- [THE ROBUSTNESS OF DIFFERENTIABLE CAUSAL DISCOVERY IN MISSPECIFIED SCENARIOS](https://openreview.net/pdf?id=iaP7yHRq1l)
- [Causal Graph Transformer for Treatment Effect Estimation Under Unknown Interference](https://openreview.net/pdf?id=foQ4AeEGG7)
- [Recovery of Causal Graph Involving Latent Variables via Homologous Surrogates](https://openreview.net/pdf?id=fGhr39bqZa)
- [A Meta-Learning Approach to Bayesian Causal Discovery](https://openreview.net/pdf?id=eeJz7eDWKO)
- [Standardizing Structural Causal Models](https://openreview.net/pdf?id=aXuWowhIYt)
- [DyCAST: Learning Dynamic Causal Structure from Time Series](https://openreview.net/pdf?id=WjDjem8mWE)
- [Estimation of single-cell and tissue perturbation effect in spatial transcriptomics via Spatial Causal Disentanglement](https://openreview.net/pdf?id=Tqdsruwyac)
- [ADAM: An Embodied Causal Agent in Open-World Environments](https://openreview.net/pdf?id=Ouu3HnIVBc)
- [Signature Kernel Conditional Independence Tests in Causal Discovery for Stochastic Processes](https://openreview.net/pdf?id=Nx4PMtJ1ER)
- [Federated Granger Causality Learning For Interdependent Clients With State Space Representation](https://openreview.net/pdf?id=KTgQGXz5xj)
- [Systems with Switching Causal Relations: A Meta-Causal Perspective](https://openreview.net/pdf?id=J9VogDTa1W)
- [Causal Order: The Key to Leveraging Imperfect Experts in Causal Inference](https://openreview.net/pdf?id=9juyeCqL0u)
- [Causal Discovery via Bayesian Optimization](https://openreview.net/pdf?id=8muemqlnG3)

<a name="FL" />

## Federated Learning, Privacy, Decentralization
- [Decoupled Subgraph Federated Learning](https://openreview.net/pdf?id=v1rFkElnIn)
- [Decentralized Sporadic Federated Learning: A Unified Algorithmic Framework with Convergence Guarantees](https://openreview.net/pdf?id=cznqgb4DNv)
- [Subgraph Federated Learning for Local Generalization](https://openreview.net/pdf?id=cH65nS5sOz)
- [Federated Granger Causality Learning For Interdependent Clients With State Space Representation](https://openreview.net/pdf?id=KTgQGXz5xj)
- [Energy-based Backdoor Defense Against Federated Graph Learning](https://openreview.net/pdf?id=5Jc7r5aqHJ)
- [On the Price of Differential Privacy for Hierarchical Clustering](https://openreview.net/pdf?id=yLhJYvkKA0)
- [Temporal Heterogeneous Graph Generation with Privacy, Utility, and Efficiency](https://openreview.net/pdf?id=tj5xJInWty)


<a name="SceneGraphs" />

## Scene Graphs
- [Hydra-SGG: Hybrid Relation Assignment for One-stage Scene Graph Generation](https://openreview.net/pdf?id=tpD1rs25Uu)
- [Interleaved Scene Graph for Interleaved Text-and-Image Generation Assessment](https://openreview.net/pdf?id=rDLgnYLM5b)
- [LASER: A Neuro-Symbolic Framework for Learning Spatio-Temporal Scene Graphs with Weak Supervision](https://openreview.net/pdf?id=HEXtydywnE)


<a name="Efficiency" />

## Graphs, GNNs and Efficiency
- [Temporal Heterogeneous Graph Generation with Privacy, Utility, and Efficiency](https://openreview.net/pdf?id=tj5xJInWty)
- [Near-Optimal Online Learning for Multi-Agent Submodular Coordination: Tight Approximation and Communication Efficiency](https://openreview.net/pdf?id=i8dYPGdB1C)
- [Conformal Generative Modeling with Improved Sample Efficiency through Sequential Greedy Filtering](https://openreview.net/pdf?id=1i6lkavJ94)
- [Synthesizing Realistic fMRI: A Physiological Dynamics-Driven Hierarchical Diffusion Model for Efficient fMRI Acquisition](https://openreview.net/pdf?id=zZ6TT254Np)
- [Pacmann: Efficient Private Approximate Nearest Neighbor Search](https://openreview.net/pdf?id=yQcFniousM)
- [Structural-Entropy-Based Sample Selection for Efficient and Effective Learning](https://openreview.net/pdf?id=xUMI52rrW7)
- [Which Tasks Should Be Compressed Together? A Causal Discovery Approach for Efficient Multi-Task Representation Compression](https://openreview.net/pdf?id=x33vSZUg0A)
- [A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation](https://openreview.net/pdf?id=wryFCrWB0A)
- [Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning](https://openreview.net/pdf?id=t9lS1lX9FQ)
- [U-Nets as Belief Propagation: Efficient Classification, Denoising, and Diffusion in Generative Hierarchical Models](https://openreview.net/pdf?id=sy1lbQxj9J)
- [CipherPrune:  Efficient and Scalable Private Transformer Inference](https://openreview.net/pdf?id=mUMvr33FTu)
- [Is uniform expressivity too restrictive? Towards efficient expressivity of GNNs](https://openreview.net/pdf?id=lsvGqR6OTf)
- [DeepGate4: Efficient and Effective Representation Learning for Circuit Design at Scale](https://openreview.net/pdf?id=b10lRabU9W)
- [Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation](https://openreview.net/pdf?id=Q0zmmNNePz)
- [Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional, Black-box Systems](https://openreview.net/pdf?id=PLskiLUBDW)
- [Fast Direct: Query-Efficient  Online Black-box Guidance  for Diffusion-model Target Generation](https://openreview.net/pdf?id=OmpTdjl7RV)
- [A Graph Enhanced Symbolic Discovery Framework For Efficient Circuit Synthesis](https://openreview.net/pdf?id=EG9nDN3eGB)
- [Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model](https://openreview.net/pdf?id=D10yarGQNk)
- [Learning Efficient Positional Encodings with Graph Neural Networks](https://openreview.net/pdf?id=AWg2tkbydO)
- [An Efficient Framework for Crediting Data Contributors of Diffusion Models](https://openreview.net/pdf?id=9EqQC2ct4H)
- [GotenNet: Rethinking Efficient 3D Equivariant Graph Neural Networks](https://openreview.net/pdf?id=5wxCQDtbMo)
- [Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition](https://openreview.net/pdf?id=41HlN8XYM5)
- [PharmacoMatch: Efficient 3D Pharmacophore Screening via Neural Subgraph Matching](https://openreview.net/pdf?id=27Qk18IZum)


# Others
- [Uncertainty Modeling in Graph Neural Networks via Stochastic Differential Equations](https://openreview.net/pdf?id=TYSQYx9vwd)
- [SurFhead: Affine Rig Blending for Geometrically Accurate 2D Gaussian Surfel Head Avatars](https://openreview.net/pdf?id=1x1gGg49jr)
- [Pushing the Limits of All-Atom Geometric Graph Neural Networks: Pre-Training, Scaling, and Zero-Shot Transfer](https://openreview.net/pdf?id=4S2L519nIX)
- [Forming Scalable, Convergent GNN Layers that Minimize a Sampling-Based Energy](https://openreview.net/pdf?id=Gq7RDMeZi4)
- [Graph-based Document Structure Analysis](https://openreview.net/pdf?id=Fu0aggezN9)
- [Matcha: Mitigating Graph Structure Shifts with Test-Time Adaptation](https://openreview.net/pdf?id=EpgoFFUM2q)
- [GOFA: A Generative One-For-All Model for Joint Graph Language Modeling](https://openreview.net/pdf?id=mIjblC9hfm)
- [Learning Graph Invariance by Harnessing Spuriosity](https://openreview.net/pdf?id=UsVJlgD1F7)
- [Credit-based self organizing maps: training deep topographic networks with minimal performance degradation](https://openreview.net/pdf?id=wMgr7wBuUo)
- [Is Graph Convolution Always Beneficial For Every Feature?](https://openreview.net/pdf?id=I9omfcWfMp)
- [GPromptShield: Elevating Resilience in Graph Prompt Tuning Against Adversarial Attacks](https://openreview.net/pdf?id=yCN4yI6zhH)
- [GraphBridge: Towards Arbitrary Transfer Learning in GNNs](https://openreview.net/pdf?id=gjRhw5S3A4)
- [Revisiting Random Walks for Learning on Graphs](https://openreview.net/pdf?id=SG1R2H3fa1)
- [Systematic Relational Reasoning With Epistemic Graph Neural Networks](https://openreview.net/pdf?id=qNp86ByQlN)
- [Greener GRASS: Enhancing GNNs with Encoding, Rewiring, and Attention](https://openreview.net/pdf?id=rEQqBZIz49)
- [Graph Neural Networks Can (Often) Count Substructures](https://openreview.net/pdf?id=sZQRUrvLn4)
- [TopoNets: High performing vision and language models with brain-like topography](https://openreview.net/pdf?id=THqWPzL00e)
- [When Graph Neural Networks Meet Dynamic Mode Decomposition](https://openreview.net/pdf?id=duGygkA3QR)
- [Beyond Sequence: Impact of Geometric Context for RNA Property Prediction](https://openreview.net/pdf?id=9htTvHkUhh)
- [Invariant Graphon Networks: Approximation and Cut Distance](https://openreview.net/pdf?id=SjufxrSOYd)
- [GNNs Getting ComFy: Community and Feature Similarity Guided Rewiring](https://openreview.net/pdf?id=g6v09VxgFw)
- [Diffusing to the Top: Boost Graph Neural Networks with Minimal Hyperparameter Tuning](https://openreview.net/pdf?id=D756s2YQ6b)
- [On the Benefits of Attribute-Driven Graph Domain Adaptation](https://openreview.net/pdf?id=t2TUw5nJsW)
- [SpaceGNN: Multi-Space Graph Neural Network for Node Anomaly Detection with Extremely Limited Labels](https://openreview.net/pdf?id=Syt4fWwVm1)
- [KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks](https://openreview.net/pdf?id=atXCzVSXTJ)
- [Joint Graph Rewiring and Feature Denoising via Spectral Resonance](https://openreview.net/pdf?id=zBbZ2vdLzH)
- [SecureGS: Boosting the Security and Fidelity of 3D Gaussian Splatting Steganography](https://openreview.net/pdf?id=H4FSx06FCZ)
- [On the Holder Stability of Multiset and Graph Neural Networks](https://openreview.net/pdf?id=P7KIGdgW8S)
- [Holographic Node Representations: Pre-training Task-Agnostic Node Embeddings](https://openreview.net/pdf?id=tGYFikNONB)
- [Rethinking Graph Neural Networks From A Geometric Perspective Of Node Features](https://openreview.net/pdf?id=lBMRmw59Lk)
- [CryoGEN: Cryogenic Electron Tomography Reconstruction via Generative Energy Nets](https://openreview.net/pdf?id=uOb7rij7sR)
- [Bringing NeRFs to the Latent Space: Inverse Graphics Autoencoder](https://openreview.net/pdf?id=LTDtjrv02Y)
- [GRAIN: Exact Graph Reconstruction from Gradients](https://openreview.net/pdf?id=7bAjVh3CG3)
- [Shapley-Guided Utility Learning for Effective Graph Inference Data Valuation](https://openreview.net/pdf?id=8X74NZpARg)
- [On the Completeness of Invariant Geometric Deep Learning Models](https://openreview.net/pdf?id=52x04chyQs)
- [Beyond Random Masking: When Dropout meets Graph Convolutional Networks](https://openreview.net/pdf?id=PwxYoMvmvy)
- [MaxCutPool: differentiable feature-aware Maxcut for pooling in graph neural networks](https://openreview.net/pdf?id=xlbXRJ2XCP)
- [Towards a Complete Logical Framework for GNN Expressiveness](https://openreview.net/pdf?id=pqOjj90Vwp)
- [Robustness Inspired Graph Backdoor Defense](https://openreview.net/pdf?id=trKNi4IUiP)
- [GLoRa: A Benchmark to Evaluate the Ability to Learn Long-Range Dependencies in Graphs](https://openreview.net/pdf?id=2jf5x5XoYk)
- [Factor Graph-based Interpretable Neural Networks](https://openreview.net/pdf?id=10DtLPsdro)
- [IGL-Bench: Establishing the Comprehensive Benchmark for Imbalanced Graph Learning](https://openreview.net/pdf?id=uTqnyF0JNR)
- [BANGS: Game-theoretic Node Selection for Graph Self-Training](https://openreview.net/pdf?id=h51mpl8Tyx)
- [Learning Diagrams: A Graphical Language for Compositional Training Regimes](https://openreview.net/pdf?id=dqyuCsBvn9)
- [Near-Optimal Policy Identification in Robust Constrained Markov Decision Processes via Epigraph Form](https://openreview.net/pdf?id=G5sPv4KSjR)
- [Node-Time Conditional Prompt Learning in Dynamic Graphs](https://openreview.net/pdf?id=kVlfYvIqaK)
- [Edge Prompt Tuning for Graph Neural Networks](https://openreview.net/pdf?id=92vMaHotTM)
- [Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems](https://openreview.net/pdf?id=Tkkrm3pA35)
- [Precedence-Constrained Winter Value for Effective Graph Data Valuation](https://openreview.net/pdf?id=tVRVE0OAyb)
- [Graph Sparsification via Mixture of Graphs](https://openreview.net/pdf?id=7ANDviElAo)
- [Bonsai: Gradient-free Graph Distillation for Node Classification](https://openreview.net/pdf?id=5x88lQ2MsH)
- [Graph Neural Networks Are More Than Filters: Revisiting and Benchmarking from A Spectral Perspective](https://openreview.net/pdf?id=nWdQX5hOL9)
- [The Effectiveness of Curvature-Based Rewiring and the Role of Hyperparameters in GNNs Revisited](https://openreview.net/pdf?id=EcrdmRT99M)
- [Training One-Dimensional Graph Neural Networks is NP-Hard](https://openreview.net/pdf?id=7BESdFZ7YA)
- [Spectro-Riemannian Graph Neural Networks](https://openreview.net/pdf?id=2MLvV7fvAz)
- [Learnable Expansion of Graph Operators for Multi-Modal Feature Fusion](https://openreview.net/pdf?id=SMZqIOSdlN)
- [ContextGNN: Beyond Two-Tower Recommendation Systems](https://openreview.net/pdf?id=nzOD1we8Z4)
- [DUALFormer: A Dual Graph Convolution and Attention Network for Node Classification](https://openreview.net/pdf?id=4v4RcAODj9)
- [Graph Neural Ricci Flow: Evolving Feature from a Curvature Perspective](https://openreview.net/pdf?id=7b2JrzdLhA)
- [Exact Certification of (Graph) Neural Networks Against Label Poisoning](https://openreview.net/pdf?id=d9aWa875kj)
- [HG-Adapter: Improving Pre-Trained Heterogeneous Graph Neural Networks with Dual Adapters](https://openreview.net/pdf?id=AEglX9CHFN)
- [Accurate and Scalable Graph Neural Networks via Message Invariance](https://openreview.net/pdf?id=UqrFPhcmFp)
- [InstantSplamp: Fast and Generalizable Stenography Framework for Generative Gaussian Splatting](https://openreview.net/pdf?id=xvhV3LvYTc)
- [Homomorphism Counts as Structural Encodings for Graph Learning](https://openreview.net/pdf?id=qFw2RFJS5g)
- [Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks](https://openreview.net/pdf?id=9tKC0YM8sX)
- [Biologically Plausible Brain Graph Transformer](https://openreview.net/pdf?id=rQyg6MnsDb)
- [GETS: Ensemble Temperature Scaling for Calibration in Graph Neural Networks](https://openreview.net/pdf?id=qgsXsqahMq)
- [Neural Multi-Objective Combinatorial Optimization via Graph-Image Multimodal Fusion](https://openreview.net/pdf?id=4sJ2FYE65U)
- [What Are Good Positional Encodings for Directed Graphs?](https://openreview.net/pdf?id=s4Wm71LFK4)
- [Centrality-guided Pre-training for Graph](https://openreview.net/pdf?id=X8E65IxA73)
- [Scalable and Certifiable Graph Unlearning: Overcoming the Approximation Error Barrier](https://openreview.net/pdf?id=pPyJyeLriR)
- [A Large-scale Training Paradigm for Graph Generative Models](https://openreview.net/pdf?id=c01YB8pF0s)
- [Valid Conformal Prediction for Dynamic GNNs](https://openreview.net/pdf?id=i3T0wvQDKg)
- [Improving Graph Neural Networks by Learning Continuous Edge Directions](https://openreview.net/pdf?id=iAmR7FfMmq)
- [Learning Structured Universe Graph with Outlier OOD Detection for Partial Matching](https://openreview.net/pdf?id=dmjQLHufev)
- [Explanations of GNN on Evolving Graphs via Axiomatic  Layer edges](https://openreview.net/pdf?id=pXN8T5RwNN)
- [AutoG: Towards automatic graph construction from tabular data](https://openreview.net/pdf?id=hovDbX4Gh6)
- [Fully-inductive Node Classification on Arbitrary Graphs](https://openreview.net/pdf?id=1Qpt43cqhg)
- [The Computational Complexity of Positive Non-Clashing Teaching in Graphs](https://openreview.net/pdf?id=Jd3Vd7GCyq)
- [Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks](https://openreview.net/pdf?id=03EkqSCKuO)
- [Quality Measures for Dynamic Graph Generative Models](https://openreview.net/pdf?id=8bjspmAMBk)
- [Open-Set Graph Anomaly Detection via Normal Structure Regularisation](https://openreview.net/pdf?id=kSvoX0xdlO)
- [ST-GCond: Self-supervised and Transferable Graph Dataset Condensation](https://openreview.net/pdf?id=wYWJFLQov9)
- [Rationalizing and Augmenting Dynamic Graph Neural Networks](https://openreview.net/pdf?id=thV5KRQFgQ)
- [Towards Continuous Reuse of Graph Models via Holistic Memory Diversification](https://openreview.net/pdf?id=Pbz4i7B0B4)
- [From GNNs to Trees: Multi-Granular Interpretability for Graph Neural Networks](https://openreview.net/pdf?id=KEUPk0wXXe)
- [PolyhedronNet: Representation Learning for Polyhedra with Surface-attributed Graph](https://openreview.net/pdf?id=BpyHIrpUOL)
- [GOttack: Universal Adversarial Attacks on Graph Neural Networks via Graph Orbits Learning](https://openreview.net/pdf?id=YbURbViE7l)
- [Graph-Guided Scene Reconstruction from Images with 3D Gaussian Splatting](https://openreview.net/pdf?id=56vHbnk35S)
- [Learning Geometric Reasoning Networks For Robot Task And Motion Planning](https://openreview.net/pdf?id=ajxAJ8GUX4)
- [Linear Transformer Topological Masking with Graph Random Features](https://openreview.net/pdf?id=6MBqQLp17E)
- [Scale-Free Graph-Language Models](https://openreview.net/pdf?id=nFcgay1Yo9)
- [Learning Long Range Dependencies on Graphs via Random Walks](https://openreview.net/pdf?id=kJ5H7oGT2M)


<a name="more" />

## More Possible Works
*(Needs Verification Yet)*

- [MANTRA: The Manifold Triangulations Assemblage](https://openreview.net/pdf?id=X6y5CC44HM)
- [CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design](https://openreview.net/pdf?id=rbnf7oe6JQ)
- [DICE: Data Influence Cascade in Decentralized Learning](https://openreview.net/pdf?id=2TIYkqieKw)
- [Integrating Protein Dynamics into Structure-Based Drug Design via Full-Atom Stochastic Flows](https://openreview.net/pdf?id=9qS3HzSDNv)
- [VLMaterial: Procedural Material Generation with Large Vision-Language Models](https://openreview.net/pdf?id=wHebuIb6IH)
- [TFG-Flow: Training-free Guidance in Multimodal Generative Flow](https://openreview.net/pdf?id=GK5ni7tIHp)
- [CapeX: Category-Agnostic Pose Estimation from Textual Point Explanation](https://openreview.net/pdf?id=scKAXgonmq)
- [Bridging the Data Provenance Gap Across Text, Speech, and Video](https://openreview.net/pdf?id=G5DziesYxL)
- [What Makes a Maze Look Like a Maze?](https://openreview.net/pdf?id=Iz75SDbRmm)
- [Neural Interactive Proofs](https://openreview.net/pdf?id=R2834dhBlo)
- [How Learnable Grids Recover Fine Detail in Low Dimesions: A Neural Tangent Kernel Analysis of Multigrid Parameteric Encodings](https://openreview.net/pdf?id=Ge7okBGZYi)
- [General Scene Adaptation for Vision-and-Language Navigation](https://openreview.net/pdf?id=2oKkQTyfz7)
- [OMG: Opacity Matters in Material Modeling with Gaussian Splatting](https://openreview.net/pdf?id=oeP6OL7ouB)
- [KLay: Accelerating Arithmetic Circuits for Neurosymbolic AI](https://openreview.net/pdf?id=Zes7Wyif8G)
- [Language Model Alignment in Multilingual Trolley Problems](https://openreview.net/pdf?id=VEqPDZIDAh)
- [Demystifying Topological Message-Passing with Relational Structures: A Case Study on Oversquashing in Simplicial Message-Passing](https://openreview.net/pdf?id=QC2qE1tcmd)
- [ThermalGaussian: Thermal 3D Gaussian Splatting](https://openreview.net/pdf?id=ybFRoGxZjs)
- [BaB-ND: Long-Horizon Motion Planning with Branch-and-Bound and Neural Dynamics](https://openreview.net/pdf?id=JXKFPJe0NU)
- [Discrete GCBF Proximal Policy Optimization for Multi-agent Safe Optimal Control](https://openreview.net/pdf?id=1X1R7P6yzt)
- [How to Verify Any (Reasonable) Distribution Property: Computationally Sound Argument Systems for Distributions](https://openreview.net/pdf?id=GfXMTAJaxZ)
- [Clique Number Estimation via Differentiable Functions of Adjacency Matrix Permutations](https://openreview.net/pdf?id=DFSb67ksVr)
- [Speech Robust Bench: A Robustness Benchmark For Speech Recognition](https://openreview.net/pdf?id=D0LuQNZfEl)
- [Fast and Accurate Blind Flexible Docking](https://openreview.net/pdf?id=iezDdA9oeB)
- [Century: A Framework and Dataset for Evaluating Historical Contextualisation of Sensitive Images](https://openreview.net/pdf?id=1KLBvrYz3V)
- [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://openreview.net/pdf?id=IjbXZdugdj)
- [CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning](https://openreview.net/pdf?id=Fg0eo2AkST)
- [Recovering Manifold Structure Using Ollivier Ricci Curvature](https://openreview.net/pdf?id=aX7X9z3vQS)
- [Latent Bayesian Optimization via Autoregressive Normalizing Flows](https://openreview.net/pdf?id=ZCOwwRAaEl)
- [In-Context Learning of Representations](https://openreview.net/pdf?id=pXlmOmlHJZ)
- [IDIV: Intrinsic Decomposition for Arbitrary Number of Input Views and Illuminations](https://openreview.net/pdf?id=uuef1HP6X7)
- [Generative Flows on Synthetic Pathway for Drug Design](https://openreview.net/pdf?id=pB1XSj2y4X)
- [CREIMBO: Cross-Regional Ensemble Interactions in Multi-view Brain Observations](https://openreview.net/pdf?id=28abpUEICJ)
- [Benchmarking Agentic Workflow Generation](https://openreview.net/pdf?id=vunPXOFmoi)
- [GlycanML: A Multi-Task and Multi-Structure Benchmark for Glycan Machine Learning](https://openreview.net/pdf?id=owEQ0FTfVj)
- [Learning to Select Nodes in Branch and Bound with Sufficient Tree Representation](https://openreview.net/pdf?id=gyvYKLEm8t)
- [Size-Generalizable RNA Structure Evaluation by Exploring Hierarchical Geometries](https://openreview.net/pdf?id=QaTBHSqmH9)
- [How Low Can You Go? Searching for the Intrinsic Dimensionality of Complex Networks using Metric Node Embeddings](https://openreview.net/pdf?id=V71ITh2w40)
- [SINGAPO: Single Image Controlled Generation of Articulated Parts in Objects](https://openreview.net/pdf?id=OdMqKszKSd)
- [OmniRe: Omni Urban Scene Reconstruction](https://openreview.net/pdf?id=11xgiMEI5o)
- [A Large-scale Dataset and Benchmark for Commuting Origin-Destination Flow Generation](https://openreview.net/pdf?id=WeJEidTzff)
- [Conformal Language Model Reasoning with Coherent Factuality](https://openreview.net/pdf?id=AJpUZd8Clb)
- [Topological Schrodinger Bridge Matching](https://openreview.net/pdf?id=WzCEiBILHu)
- [A Soft and Fast Pattern Matcher for Billion-Scale Corpus Searches](https://openreview.net/pdf?id=Q6PAnqYVpo)
- [KinPFN: Bayesian Approximation of RNA Folding Kinetics using Prior-Data Fitted Networks](https://openreview.net/pdf?id=E1m5yGMOiV)
- [Extendable and Iterative Structure Learning for Bayesian Networks](https://openreview.net/pdf?id=3n6DYH3cIP)
- [Language Representations Can be What Recommenders Need: Findings and Potentials](https://openreview.net/pdf?id=eIJfOIMN9z)
- [Optimizing 4D Gaussians for Dynamic Scene Video from Single Landscape Images](https://openreview.net/pdf?id=IcYDRzcccP)
- [Multi-domain Distribution Learning for De Novo Drug Design](https://openreview.net/pdf?id=g3VCIM94ke)
- [Distribution-Specific Agnostic Conditional Classification With Halfspaces](https://openreview.net/pdf?id=KZEqbwJfTl)
- [Inverse Rendering for Shape, Light, and Material Decomposition using Multi-Bounce Path Tracing and Reservoir Sampling](https://openreview.net/pdf?id=KEXoZxTwbr)
- [UniCoTT: A Unified Framework for Structural Chain-of-Thought Distillation](https://openreview.net/pdf?id=3baOKeI2EU)
- [Balanced Ranking with Relative Centrality: A multi-core periphery perspective](https://openreview.net/pdf?id=21rSeWJHPF)
- [BTBS-LNS: Binarized-Tightening, Branch and Search on Learning LNS Policies for MIP](https://openreview.net/pdf?id=siHHqDDzvS)
- [SleepSMC: Ubiquitous Sleep Staging via Supervised Multimodal Coordination](https://openreview.net/pdf?id=B5VEi5d3p2)
- [GeSubNet: Gene Interaction Inference for Disease Subtype Network Generation](https://openreview.net/pdf?id=ja4rpheN2n)
- [Semialgebraic Neural Networks: From roots to representations](https://openreview.net/pdf?id=zboCXnuNv7)
- [ReSi: A Comprehensive Benchmark for Representational Similarity Measures](https://openreview.net/pdf?id=PRvdO3nfFi)
- [SIM: Surface-based fMRI Analysis for Inter-Subject Multimodal Decoding from Movie-Watching Experiments](https://openreview.net/pdf?id=OJsMGsO6yn)
- [Inverse Constitutional AI: Compressing Preferences into Principles](https://openreview.net/pdf?id=9FRwkPw3Cn)
- [b"Schur's Positive-Definite Network: Deep Learning in the SPD cone with structure"](https://openreview.net/pdf?id=v1B4aet9ct)
- [GOAL: A Generalist Combinatorial Optimization Agent Learner](https://openreview.net/pdf?id=z2z9suDRjw)
- [The "Law\'\' of the Unconscious Contrastive Learner: Probabilistic Alignment of Unpaired Modalities](https://openreview.net/pdf?id=DsIOUoZkVk)
- [SMI-Editor: Edit-based SMILES Language Model with Fragment-level Supervision](https://openreview.net/pdf?id=M29nUGozPa)
- [Rethinking the role of frames for SE(3)-invariant crystal structure modeling](https://openreview.net/pdf?id=gzxDjnvBDa)
- [PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training](https://openreview.net/pdf?id=j4LITBSUjs)
- [Vector-ICL: In-context Learning with Continuous Vector Representations](https://openreview.net/pdf?id=xing7dDGh3)
- [Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts](https://openreview.net/pdf?id=NJxCpMt0sf)
- [Reframing Structure-Based Drug Design Model Evaluation via Metrics Correlated to Practical Needs](https://openreview.net/pdf?id=RyWypcIMiE)
- [LiveXiv - A Multi-Modal live benchmark based on Arxiv papers content](https://openreview.net/pdf?id=SulRfnEVK4)
- [Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Monocular Videos](https://openreview.net/pdf?id=LuGHbK8qTa)
- [Towards Homogeneous Lexical Tone Decoding from Heterogeneous Intracranial Recordings](https://openreview.net/pdf?id=cWEfRkYj46)
- [ProtoSnap: Prototype Alignment For Cuneiform Signs](https://openreview.net/pdf?id=XHTirKsQV6)
- [Exposure Bracketing Is All You Need For A High-Quality Image](https://openreview.net/pdf?id=rDIf6NA5mj)
- [AtomSurf: Surface Representation for Learning on Protein Structures](https://openreview.net/pdf?id=ARQIJXFcTH)
- [Optimal Flow Transport and its Entropic Regularization: a GPU-friendly Matrix Iterative Algorithm for Flow Balance Satisfaction](https://openreview.net/pdf?id=NtSlKEJ2DS)
- [Understanding Virtual Nodes: Oversquashing and Node Heterogeneity](https://openreview.net/pdf?id=NmcOAwRyH5)
- [First-Person Fairness in Chatbots](https://openreview.net/pdf?id=TlAdgeoDTo)
- [DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models](https://openreview.net/pdf?id=VOAMTA8jKu)
- [3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting](https://openreview.net/pdf?id=iKDbLpVgQc)
- [Decoupling Layout from Glyph in Online Chinese Handwriting Generation](https://openreview.net/pdf?id=DhHIw9Nbl1)
- [HGM3: Hierarchical Generative Masked Motion Modeling with Hard Token Mining](https://openreview.net/pdf?id=IEul1M5pyk)
- [Towards Faster Decentralized Stochastic Optimization with Communication Compression](https://openreview.net/pdf?id=CMMpcs9prj)
- [Operator Deep Smoothing for Implied Volatility](https://openreview.net/pdf?id=DPlUWG4WMw)
- [Tree of Attributes Prompt Learning for Vision-Language Models](https://openreview.net/pdf?id=wFs2E5wCw6)
- [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://openreview.net/pdf?id=NPNUHgHF2w)
- [A Generic Framework for Conformal Fairness](https://openreview.net/pdf?id=xiQNfYl33p)
- [ECHOPulse: ECG Controlled Echocardio-gram Video Generation](https://openreview.net/pdf?id=i2r7LDjba3)
- [Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks](https://openreview.net/pdf?id=4NTrco82W0)
- [Accelerating Training with Neuron Interaction and Nowcasting Networks](https://openreview.net/pdf?id=cUFIil6hEG)
- [Streaming Algorithms For $\\ell_p$ Flows and $\\ell_p$ Regression](https://openreview.net/pdf?id=Kpjvm2mB0K)
- [Dissecting Adversarial Robustness of Multimodal LM Agents](https://openreview.net/pdf?id=YauQYh2k1g)
- [Mixture of Parrots: Experts improve memorization more than reasoning](https://openreview.net/pdf?id=9XETcRsufZ)
- [CryoFM: A Flow-based Foundation Model for Cryo-EM Densities](https://openreview.net/pdf?id=T4sMzjy7fO)
- [Breaking Free from MMI: A New Frontier in Rationalization by Probing Input Utilization](https://openreview.net/pdf?id=WZ0s2smcKP)
- [NNsight and NDIF: Democratizing Access to Foundation Model Internals](https://openreview.net/pdf?id=MxbEiFRf39)
- [Adversarial Latent Feature Augmentation for Fairness](https://openreview.net/pdf?id=cNaHOdvh9J)
- [AnalogGenie: A Generative Engine for Automatic Discovery of Analog Circuit Topologies](https://openreview.net/pdf?id=jCPak79Kev)
- [Neural Spacetimes for DAG Representation Learning](https://openreview.net/pdf?id=skGSOcrIj7)
- [Multiple Heads are Better than One: Mixture of Modality Knowledge Experts for Entity Representation Learning](https://openreview.net/pdf?id=ue1Tt3h1VC)
- [ReGen: Generative Robot Simulation via Inverse Design](https://openreview.net/pdf?id=EbCUbPZjM1)
- [Variance-Reducing Couplings for Random Features](https://openreview.net/pdf?id=oJLpXraSLb)
- [CLIBD: Bridging Vision and Genomics for Biodiversity Monitoring at Scale](https://openreview.net/pdf?id=d5HUnyByAI)
- [Node Similarities under Random Projections: Limits and Pathological Cases](https://openreview.net/pdf?id=Frok9AItud)
- [Vision CNNs trained to estimate spatial latents learned similar ventral-stream-aligned representations](https://openreview.net/pdf?id=emMMa4q0qw)
- [Multimodal Lego: Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine](https://openreview.net/pdf?id=pH543jrbe8)
- [Contextualizing biological perturbation experiments through language](https://openreview.net/pdf?id=5WEpbilssv)
- [Doubly robust identification of treatment effects from multiple environments](https://openreview.net/pdf?id=9vTAkJ9Tik)
- [OSCAR: Operating System Control via State-Aware Reasoning and Re-Planning](https://openreview.net/pdf?id=VuTrZzrPfn)
- [Integrative Decoding: Improving Factuality via Implicit Self-consistency](https://openreview.net/pdf?id=gGWYecsK1U)
- [Training Free Guided Flow-Matching with Optimal Control](https://openreview.net/pdf?id=61ss5RA1MM)
- [Conformal Structured Prediction](https://openreview.net/pdf?id=2ATD8a8P3C)
- [MindSearch: Mimicking Human Minds Elicits Deep AI Searcher](https://openreview.net/pdf?id=xgtXkyqw1f)
- [Circuit Representation Learning with Masked Gate Modeling and Verilog-AIG Alignment](https://openreview.net/pdf?id=US9k5TXVLZ)
- [ToolDial: Multi-turn Dialogue Generation Method for Tool-Augmented Language Models](https://openreview.net/pdf?id=J1J5eGJsKZ)
- [Controllable Satellite-to-Street-View Synthesis with Precise Pose Alignment and Zero-Shot Environmental Control](https://openreview.net/pdf?id=f92M45YRfh)
- [An Exploration with Entropy Constrained 3D Gaussians for 2D Video Compression](https://openreview.net/pdf?id=JbRM5QKRDd)
- [ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://openreview.net/pdf?id=KSLkFYHlYg)
- [Intelligent Go-Explore: Standing on the Shoulders of Giant Foundation Models](https://openreview.net/pdf?id=apErWGzCAA)
- [UniMatch: Universal Matching from Atom to Task for Few-Shot Drug Discovery](https://openreview.net/pdf?id=v9EjwMM55Y)
- [Redefining the task of Bioactivity Prediction](https://openreview.net/pdf?id=S8gbnkCgxZ)
- [SimXRD-4M: Big Simulated X-ray Diffraction Data and Crystal Symmetry Classification Benchmark](https://openreview.net/pdf?id=mkuB677eMM)
- [Context Steering: Controllable Personalization at Inference Time](https://openreview.net/pdf?id=xQCXInDq0m)
- [Locality Sensitive Avatars From Video](https://openreview.net/pdf?id=SVta2eQNt3)
- [Perplexity Trap: PLM-Based Retrievers Overrate Low Perplexity Documents](https://openreview.net/pdf?id=U1T6sq12uj)
- [PaPaGei: Open Foundation Models for Optical Physiological Signals](https://openreview.net/pdf?id=kYwTmlq6Vn)
- [Eliminating Position Bias of Language Models: A Mechanistic Approach](https://openreview.net/pdf?id=fvkElsJOsN)




---


**Missing any paper?**
If any paper is absent from the list, please feel free to [mail](mailto:azminetoushik.wasi@gmail.com) or [open an issue](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024/issues/new/choose) or submit a pull request. I'll gladly add that! Also, If I mis-categorized, please knock!

---

## More Collectons:
- [Awesome **NeurIPS'24** ***Molecular ML*** Paper Collection](https://github.com/azminewasi/Awesome-MoML-NeurIPS24)
- [**Awesome NeurIPS 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-NeurIPS2024)
- [**Awesome ICML 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024)
- [**Awesome ICLR 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2024)
- [**Awesome-LLMs-ICLR-24**](https://github.com/azminewasi/Awesome-LLMs-ICLR-24/)

---

## ✨ **Credits**
**Azmine Toushik Wasi**

 [![website](https://img.shields.io/badge/-Website-blue?style=flat-square&logo=rss&color=1f1f15)](https://azminewasi.github.io) 
 [![linkedin](https://img.shields.io/badge/LinkedIn-%320beff?style=flat-square&logo=linkedin&color=1f1f18)](https://www.linkedin.com/in/azmine-toushik-wasi/) 
 [![kaggle](https://img.shields.io/badge/Kaggle-%2320beff?style=flat-square&logo=kaggle&color=1f1f1f)](https://www.kaggle.com/azminetoushikwasi) 
 [![google-scholar](https://img.shields.io/badge/Google%20Scholar-%2320beff?style=flat-square&logo=google-scholar&color=1f1f18)](https://scholar.google.com/citations?user=X3gRvogAAAAJ&hl=en) 
 [![facebook](https://img.shields.io/badge/Facebook-%2320beff?style=flat-square&logo=facebook&color=1f1f15)](https://www.facebook.com/cholche.gari.zatrabari/)
