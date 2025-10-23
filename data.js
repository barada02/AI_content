// AI Engineering Learning Data
const aiLearningData = {
    // Core learning tracks with dependencies and time estimates
    learningTracks: {
        foundations: {
            title: "Mathematical & CS Foundations",
            description: "Essential mathematical and computer science foundations for AI",
            duration: "4-6 months",
            color: "#48bb78",
            topics: [
                {
                    id: "linear-algebra",
                    title: "Linear Algebra",
                    description: "Vectors, matrices, eigenvalues, SVD - fundamental for ML",
                    duration: "3-4 weeks",
                    prerequisites: [],
                    skills: ["Mathematics", "Problem Solving"],
                    importance: "critical",
                    resources: ["MIT 18.06", "3Blue1Brown Linear Algebra"],
                    projects: ["Matrix Operations Library", "PCA Implementation"]
                },
                {
                    id: "calculus",
                    title: "Calculus & Optimization",
                    description: "Derivatives, gradients, optimization theory for neural networks",
                    duration: "3-4 weeks",
                    prerequisites: [],
                    skills: ["Mathematics", "Optimization"],
                    importance: "critical",
                    resources: ["Khan Academy Calculus", "Optimization Theory"],
                    projects: ["Gradient Descent Visualization", "Function Optimization"]
                },
                {
                    id: "probability",
                    title: "Probability & Statistics",
                    description: "Bayesian thinking, distributions, statistical inference",
                    duration: "4-5 weeks",
                    prerequisites: [],
                    skills: ["Statistics", "Bayesian Reasoning"],
                    importance: "critical",
                    resources: ["Think Stats", "Bayesian Methods for ML"],
                    projects: ["Bayesian Classifier", "Statistical Analysis Tool"]
                },
                {
                    id: "algorithms",
                    title: "Algorithms & Data Structures",
                    description: "Computational complexity, graph algorithms, dynamic programming",
                    duration: "6-8 weeks",
                    prerequisites: [],
                    skills: ["Programming", "Problem Solving"],
                    importance: "high",
                    resources: ["CLRS", "LeetCode", "AlgoExpert"],
                    projects: ["Graph Search Algorithms", "Dynamic Programming Solutions"]
                }
            ]
        },
        programming: {
            title: "Programming & Tools",
            description: "Programming languages and tools essential for AI development",
            duration: "3-4 months",
            color: "#667eea",
            topics: [
                {
                    id: "python-basics",
                    title: "Python Programming",
                    description: "Python fundamentals, OOP, functional programming",
                    duration: "4-6 weeks",
                    prerequisites: [],
                    skills: ["Python", "Programming"],
                    importance: "critical",
                    resources: ["Python.org Tutorial", "Automate the Boring Stuff"],
                    projects: ["Python CLI Tools", "Data Processing Scripts"]
                },
                {
                    id: "scientific-python",
                    title: "Scientific Python Stack",
                    description: "NumPy, Pandas, Matplotlib, Jupyter for data science",
                    duration: "3-4 weeks",
                    prerequisites: ["python-basics"],
                    skills: ["Data Science", "Visualization"],
                    importance: "critical",
                    resources: ["Python Data Science Handbook", "Pandas Documentation"],
                    projects: ["Data Analysis Pipeline", "Interactive Visualizations"]
                },
                {
                    id: "git-version-control",
                    title: "Git & Version Control",
                    description: "Version control, collaboration, open source contribution",
                    duration: "1-2 weeks",
                    prerequisites: [],
                    skills: ["Development Tools", "Collaboration"],
                    importance: "high",
                    resources: ["Pro Git Book", "GitHub Docs"],
                    projects: ["Open Source Contributions", "Project Portfolio"]
                },
                {
                    id: "cloud-platforms",
                    title: "Cloud Computing Basics",
                    description: "AWS/GCP/Azure fundamentals, containerization with Docker",
                    duration: "2-3 weeks",
                    prerequisites: ["python-basics"],
                    skills: ["Cloud Computing", "DevOps"],
                    importance: "medium",
                    resources: ["AWS ML Specialty", "Docker Documentation"],
                    projects: ["Cloud ML Pipeline", "Containerized Applications"]
                }
            ]
        },
        machineLearning: {
            title: "Core Machine Learning",
            description: "Fundamental machine learning algorithms and concepts",
            duration: "4-5 months",
            color: "#ed8936",
            topics: [
                {
                    id: "ml-fundamentals",
                    title: "ML Fundamentals",
                    description: "Supervised/unsupervised learning, bias-variance tradeoff",
                    duration: "3-4 weeks",
                    prerequisites: ["linear-algebra", "probability", "python-basics"],
                    skills: ["Machine Learning", "Data Analysis"],
                    importance: "critical",
                    resources: ["Andrew Ng Course", "Hands-On ML"],
                    projects: ["Linear Regression from Scratch", "Classification Pipeline"]
                },
                {
                    id: "classical-ml",
                    title: "Classical ML Algorithms",
                    description: "Decision trees, SVM, ensemble methods, clustering",
                    duration: "4-5 weeks",
                    prerequisites: ["ml-fundamentals"],
                    skills: ["Algorithm Implementation", "Feature Engineering"],
                    importance: "high",
                    resources: ["Scikit-learn Documentation", "Pattern Recognition"],
                    projects: ["Ensemble Methods Comparison", "Customer Segmentation"]
                },
                {
                    id: "feature-engineering",
                    title: "Feature Engineering & Selection",
                    description: "Data preprocessing, feature selection, dimensionality reduction",
                    duration: "2-3 weeks",
                    prerequisites: ["ml-fundamentals"],
                    skills: ["Data Preprocessing", "Feature Selection"],
                    importance: "high",
                    resources: ["Feature Engineering for ML", "Kaggle Learn"],
                    projects: ["Feature Selection Tool", "Dimensionality Reduction Comparison"]
                },
                {
                    id: "model-evaluation",
                    title: "Model Evaluation & Validation",
                    description: "Cross-validation, metrics, hyperparameter tuning",
                    duration: "2-3 weeks",
                    prerequisites: ["classical-ml"],
                    skills: ["Model Validation", "Hyperparameter Tuning"],
                    importance: "high",
                    resources: ["ML Yearning", "Cross-Validation Techniques"],
                    projects: ["AutoML Framework", "Model Comparison Tool"]
                }
            ]
        },
        deepLearning: {
            title: "Deep Learning",
            description: "Neural networks, deep learning architectures and applications",
            duration: "5-6 months",
            color: "#805ad5",
            topics: [
                {
                    id: "neural-networks",
                    title: "Neural Network Fundamentals",
                    description: "Perceptrons, backpropagation, activation functions",
                    duration: "3-4 weeks",
                    prerequisites: ["calculus", "ml-fundamentals"],
                    skills: ["Deep Learning", "Neural Networks"],
                    importance: "critical",
                    resources: ["Deep Learning Book", "CS231n"],
                    projects: ["Neural Network from Scratch", "Backpropagation Visualization"]
                },
                {
                    id: "cnn",
                    title: "Convolutional Neural Networks",
                    description: "Computer vision, image classification, object detection",
                    duration: "4-5 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["Computer Vision", "Image Processing"],
                    importance: "high",
                    resources: ["CS231n", "PyTorch Vision Tutorial"],
                    projects: ["Image Classifier", "Object Detection System"]
                },
                {
                    id: "rnn-lstm",
                    title: "RNNs & LSTMs",
                    description: "Sequence modeling, time series, language modeling",
                    duration: "3-4 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["Sequence Modeling", "NLP"],
                    importance: "high",
                    resources: ["Understanding LSTMs", "RNN Tutorial"],
                    projects: ["Text Generator", "Time Series Predictor"]
                },
                {
                    id: "transformers",
                    title: "Transformers & Attention",
                    description: "Attention mechanisms, transformer architecture, BERT/GPT",
                    duration: "4-6 weeks",
                    prerequisites: ["rnn-lstm"],
                    skills: ["Transformers", "Attention Mechanisms"],
                    importance: "critical",
                    resources: ["Attention Is All You Need", "Annotated Transformer"],
                    projects: ["Transformer from Scratch", "Fine-tuned Language Model"]
                },
                {
                    id: "generative-models",
                    title: "Generative Models",
                    description: "GANs, VAEs, diffusion models, generative AI",
                    duration: "4-5 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["Generative AI", "Deep Learning"],
                    importance: "high",
                    resources: ["GAN Papers", "Diffusion Models"],
                    projects: ["GAN Implementation", "Image Generation Tool"]
                }
            ]
        },
        advancedTopics: {
            title: "Advanced AI Topics",
            description: "Cutting-edge AI techniques and specialized areas",
            duration: "6-8 months",
            color: "#f56565",
            topics: [
                {
                    id: "reinforcement-learning",
                    title: "Reinforcement Learning",
                    description: "Q-learning, policy gradients, actor-critic methods",
                    duration: "5-6 weeks",
                    prerequisites: ["neural-networks", "probability"],
                    skills: ["Reinforcement Learning", "Game Theory"],
                    importance: "high",
                    resources: ["Sutton & Barto", "OpenAI Gym"],
                    projects: ["Game Playing Agent", "Robot Control System"]
                },
                {
                    id: "meta-learning",
                    title: "Meta-Learning",
                    description: "Learning to learn, few-shot learning, MAML",
                    duration: "3-4 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["Meta-Learning", "Few-Shot Learning"],
                    importance: "medium",
                    resources: ["Meta-Learning Papers", "MAML Tutorial"],
                    projects: ["Few-Shot Classifier", "Meta-Learning Framework"]
                },
                {
                    id: "neuro-symbolic",
                    title: "Neuro-Symbolic AI",
                    description: "Combining neural networks with symbolic reasoning",
                    duration: "4-5 weeks",
                    prerequisites: ["neural-networks", "algorithms"],
                    skills: ["Symbolic Reasoning", "Knowledge Graphs"],
                    importance: "medium",
                    resources: ["Neuro-Symbolic Papers", "Knowledge Graph Tutorial"],
                    projects: ["Reasoning System", "Knowledge Graph Neural Network"]
                },
                {
                    id: "multimodal-ai",
                    title: "Multimodal AI",
                    description: "Vision-language models, multimodal understanding",
                    duration: "4-5 weeks",
                    prerequisites: ["cnn", "transformers"],
                    skills: ["Multimodal Learning", "Vision-Language"],
                    importance: "high",
                    resources: ["CLIP Paper", "Multimodal Tutorials"],
                    projects: ["Vision-Language Model", "Multimodal Search Engine"]
                }
            ]
        },
        agiResearch: {
            title: "AGI Research Frontiers",
            description: "Cutting-edge research towards Artificial General Intelligence",
            duration: "Ongoing",
            color: "#38b2ac",
            topics: [
                {
                    id: "program-synthesis",
                    title: "Program Synthesis",
                    description: "Automatic program generation, code generation models",
                    duration: "4-6 weeks",
                    prerequisites: ["transformers", "algorithms"],
                    skills: ["Program Synthesis", "Code Generation"],
                    importance: "medium",
                    resources: ["Program Synthesis Papers", "CodeT5 Tutorial"],
                    projects: ["Code Generation Tool", "Program Synthesis Framework"]
                },
                {
                    id: "causal-inference",
                    title: "Causal Inference & Reasoning",
                    description: "Causal models, counterfactual reasoning, causal discovery",
                    duration: "5-6 weeks",
                    prerequisites: ["probability", "ml-fundamentals"],
                    skills: ["Causal Reasoning", "Statistical Inference"],
                    importance: "medium",
                    resources: ["The Book of Why", "Causal Inference Papers"],
                    projects: ["Causal Discovery Tool", "Counterfactual Generator"]
                },
                {
                    id: "continual-learning",
                    title: "Continual Learning",
                    description: "Lifelong learning, catastrophic forgetting, plasticity",
                    duration: "3-4 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["Continual Learning", "Memory Systems"],
                    importance: "medium",
                    resources: ["Continual Learning Papers", "EWC Tutorial"],
                    projects: ["Lifelong Learning System", "Memory Replay Network"]
                },
                {
                    id: "ai-alignment",
                    title: "AI Safety & Alignment",
                    description: "Value alignment, interpretability, robustness",
                    duration: "4-5 weeks",
                    prerequisites: ["neural-networks"],
                    skills: ["AI Safety", "Interpretability"],
                    importance: "high",
                    resources: ["AI Alignment Papers", "Interpretability Research"],
                    projects: ["Model Interpretability Tool", "Safety Verification System"]
                }
            ]
        }
    },

    // Comprehensive resource library
    resources: [
        // Books
        {
            id: "hands-on-ml",
            title: "Hands-On Machine Learning",
            type: "books",
            author: "Aurélien Géron",
            description: "Practical guide to machine learning with Scikit-Learn and TensorFlow",
            difficulty: "intermediate",
            rating: 4.8,
            url: "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"
        },
        {
            id: "deep-learning-book",
            title: "Deep Learning",
            type: "books",
            author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville",
            description: "Comprehensive theoretical foundation of deep learning",
            difficulty: "advanced",
            rating: 4.7,
            url: "https://www.deeplearningbook.org/"
        },
        {
            id: "pattern-recognition",
            title: "Pattern Recognition and Machine Learning",
            type: "books",
            author: "Christopher Bishop",
            description: "Mathematical foundations of machine learning algorithms",
            difficulty: "advanced",
            rating: 4.6,
            url: "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
        },

        // Courses
        {
            id: "andrew-ng-ml",
            title: "Machine Learning Course",
            type: "courses",
            author: "Andrew Ng",
            description: "Foundational machine learning course covering core algorithms",
            difficulty: "beginner",
            rating: 4.9,
            url: "https://www.coursera.org/learn/machine-learning"
        },
        {
            id: "cs231n",
            title: "CS231n: Deep Learning for Computer Vision",
            type: "courses",
            author: "Stanford University",
            description: "Comprehensive course on convolutional neural networks",
            difficulty: "intermediate",
            rating: 4.8,
            url: "http://cs231n.stanford.edu/"
        },
        {
            id: "fast-ai",
            title: "Practical Deep Learning for Coders",
            type: "courses",
            author: "fast.ai",
            description: "Top-down approach to deep learning with practical applications",
            difficulty: "intermediate",
            rating: 4.7,
            url: "https://course.fast.ai/"
        },

        // Research Papers
        {
            id: "attention-paper",
            title: "Attention Is All You Need",
            type: "papers",
            author: "Vaswani et al.",
            description: "Introduced the transformer architecture revolutionizing NLP",
            difficulty: "advanced",
            rating: 4.9,
            url: "https://arxiv.org/abs/1706.03762"
        },
        {
            id: "gpt-paper",
            title: "Language Models are Unsupervised Multitask Learners",
            type: "papers",
            author: "Radford et al.",
            description: "GPT-2 paper demonstrating large-scale language model capabilities",
            difficulty: "advanced",
            rating: 4.8,
            url: "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
        },

        // Tools & Frameworks
        {
            id: "pytorch",
            title: "PyTorch",
            type: "tools",
            author: "Meta AI",
            description: "Dynamic deep learning framework popular in research",
            difficulty: "intermediate",
            rating: 4.8,
            url: "https://pytorch.org/"
        },
        {
            id: "tensorflow",
            title: "TensorFlow",
            type: "tools",
            author: "Google",
            description: "Comprehensive machine learning platform",
            difficulty: "intermediate",
            rating: 4.6,
            url: "https://www.tensorflow.org/"
        },
        {
            id: "huggingface",
            title: "Hugging Face Transformers",
            type: "tools",
            author: "Hugging Face",
            description: "State-of-the-art pre-trained models and NLP tools",
            difficulty: "beginner",
            rating: 4.9,
            url: "https://huggingface.co/transformers/"
        },

        // Tutorials
        {
            id: "pytorch-tutorial",
            title: "PyTorch Official Tutorials",
            type: "tutorials",
            author: "PyTorch Team",
            description: "Official tutorials covering PyTorch fundamentals to advanced topics",
            difficulty: "beginner",
            rating: 4.7,
            url: "https://pytorch.org/tutorials/"
        }
    ],

    // Hands-on projects organized by difficulty and topic
    projects: [
        // Beginner Projects
        {
            id: "linear-regression-scratch",
            title: "Linear Regression from Scratch",
            difficulty: "beginner",
            description: "Implement linear regression using only NumPy to understand gradient descent",
            skills: ["Python", "NumPy", "Mathematics"],
            duration: "1-2 weeks",
            topics: ["ml-fundamentals"],
            learningObjectives: [
                "Understand gradient descent optimization",
                "Implement cost function and derivatives",
                "Visualize learning process"
            ],
            deliverables: [
                "Complete implementation in Python",
                "Visualization of cost function convergence",
                "Comparison with scikit-learn implementation"
            ]
        },
        {
            id: "mnist-classifier",
            title: "MNIST Digit Classification",
            difficulty: "beginner",
            description: "Build a neural network to classify handwritten digits",
            skills: ["Deep Learning", "PyTorch", "Computer Vision"],
            duration: "1-2 weeks",
            topics: ["neural-networks"],
            learningObjectives: [
                "Build and train neural networks",
                "Understand image preprocessing",
                "Evaluate model performance"
            ]
        },
        {
            id: "sentiment-analyzer",
            title: "Sentiment Analysis Tool",
            difficulty: "beginner",
            description: "Create a sentiment classifier for movie reviews",
            skills: ["NLP", "Text Processing", "Classification"],
            duration: "2 weeks",
            topics: ["ml-fundamentals", "classical-ml"],
            learningObjectives: [
                "Text preprocessing and vectorization",
                "Feature engineering for text",
                "Model evaluation for NLP"
            ]
        },

        // Intermediate Projects
        {
            id: "recommendation-system",
            title: "Movie Recommendation System",
            difficulty: "intermediate",
            description: "Build collaborative and content-based recommendation systems",
            skills: ["Collaborative Filtering", "Matrix Factorization", "Embeddings"],
            duration: "3-4 weeks",
            topics: ["classical-ml", "neural-networks"],
            learningObjectives: [
                "Implement collaborative filtering",
                "Use matrix factorization techniques",
                "Build hybrid recommendation systems"
            ]
        },
        {
            id: "gpt-clone",
            title: "Mini-GPT Implementation",
            difficulty: "intermediate",
            description: "Build a smaller version of GPT transformer model",
            skills: ["Transformers", "Attention", "Language Modeling"],
            duration: "4-5 weeks",
            topics: ["transformers"],
            learningObjectives: [
                "Implement multi-head attention",
                "Build transformer architecture",
                "Train on text generation task"
            ]
        },
        {
            id: "face-recognition",
            title: "Face Recognition System",
            difficulty: "intermediate",
            description: "Build an end-to-end face recognition pipeline",
            skills: ["Computer Vision", "CNNs", "Face Detection"],
            duration: "3-4 weeks",
            topics: ["cnn"],
            learningObjectives: [
                "Face detection and alignment",
                "Feature extraction with CNNs",
                "Similarity matching and verification"
            ]
        },

        // Advanced Projects
        {
            id: "alphago-clone",
            title: "Game Playing AI (AlphaGo Style)",
            difficulty: "advanced",
            description: "Implement MCTS with neural networks for board games",
            skills: ["Reinforcement Learning", "MCTS", "Self-Play"],
            duration: "6-8 weeks",
            topics: ["reinforcement-learning"],
            learningObjectives: [
                "Implement Monte Carlo Tree Search",
                "Combine neural networks with tree search",
                "Self-play training methodology"
            ]
        },
        {
            id: "multimodal-search",
            title: "Multimodal Search Engine",
            difficulty: "advanced",
            description: "Search images using text queries and vice versa",
            skills: ["Multimodal Learning", "Embeddings", "CLIP"],
            duration: "5-6 weeks",
            topics: ["multimodal-ai"],
            learningObjectives: [
                "Align visual and textual representations",
                "Build cross-modal retrieval system",
                "Implement similarity search"
            ]
        },
        {
            id: "neural-architecture-search",
            title: "Neural Architecture Search",
            difficulty: "advanced",
            description: "Automatically discover optimal neural network architectures",
            skills: ["AutoML", "Architecture Search", "Optimization"],
            duration: "6-8 weeks",
            topics: ["meta-learning"],
            learningObjectives: [
                "Implement differentiable architecture search",
                "Optimize network topology",
                "Evaluate architectural innovations"
            ]
        },

        // Research Projects
        {
            id: "program-synthesizer",
            title: "AI Code Generator",
            difficulty: "research",
            description: "Build a system that generates code from natural language descriptions",
            skills: ["Program Synthesis", "Code Generation", "Transformers"],
            duration: "8-10 weeks",
            topics: ["program-synthesis", "transformers"],
            learningObjectives: [
                "Understand program synthesis techniques",
                "Train code generation models",
                "Implement execution-guided synthesis"
            ]
        },
        {
            id: "causal-discovery",
            title: "Causal Structure Discovery",
            difficulty: "research",
            description: "Automatically discover causal relationships from observational data",
            skills: ["Causal Inference", "Graph Learning", "Statistics"],
            duration: "8-12 weeks",
            topics: ["causal-inference"],
            learningObjectives: [
                "Implement causal discovery algorithms",
                "Handle confounding variables",
                "Validate causal hypotheses"
            ]
        },
        {
            id: "continual-learning-system",
            title: "Lifelong Learning Agent",
            difficulty: "research",
            description: "Build an AI system that learns continuously without forgetting",
            skills: ["Continual Learning", "Memory Systems", "Plasticity"],
            duration: "10-12 weeks",
            topics: ["continual-learning"],
            learningObjectives: [
                "Implement memory replay mechanisms",
                "Handle catastrophic forgetting",
                "Design adaptive learning systems"
            ]
        }
    ],

    // Learning milestones and achievement system
    milestones: [
        {
            id: "foundation-complete",
            title: "Foundations Master",
            description: "Completed all mathematical and programming fundamentals",
            requirements: ["linear-algebra", "calculus", "probability", "python-basics", "scientific-python"],
            estimatedDate: "Month 3",
            status: "upcoming"
        },
        {
            id: "ml-practitioner",
            title: "ML Practitioner",
            description: "Can build and deploy classical machine learning models",
            requirements: ["ml-fundamentals", "classical-ml", "feature-engineering", "model-evaluation"],
            estimatedDate: "Month 6",
            status: "upcoming"
        },
        {
            id: "dl-engineer",
            title: "Deep Learning Engineer",
            description: "Proficient in neural networks and deep learning architectures",
            requirements: ["neural-networks", "cnn", "rnn-lstm", "transformers"],
            estimatedDate: "Month 12",
            status: "upcoming"
        },
        {
            id: "ai-researcher",
            title: "AI Researcher",
            description: "Can contribute to cutting-edge AI research",
            requirements: ["reinforcement-learning", "meta-learning", "neuro-symbolic", "multimodal-ai"],
            estimatedDate: "Month 18",
            status: "upcoming"
        },
        {
            id: "agi-pioneer",
            title: "AGI Pioneer",
            description: "Working on the frontiers of artificial general intelligence",
            requirements: ["program-synthesis", "causal-inference", "continual-learning", "ai-alignment"],
            estimatedDate: "Month 24+",
            status: "upcoming"
        }
    ],

    // Knowledge graph for visualizing dependencies
    knowledgeGraph: {
        nodes: [
            // Foundation nodes
            { id: "linear-algebra", group: "math", level: 1 },
            { id: "calculus", group: "math", level: 1 },
            { id: "probability", group: "math", level: 1 },
            { id: "algorithms", group: "cs", level: 1 },
            { id: "python-basics", group: "programming", level: 1 },
            
            // Intermediate nodes
            { id: "scientific-python", group: "programming", level: 2 },
            { id: "ml-fundamentals", group: "ml", level: 2 },
            { id: "neural-networks", group: "dl", level: 2 },
            
            // Advanced nodes
            { id: "transformers", group: "dl", level: 3 },
            { id: "reinforcement-learning", group: "advanced", level: 3 },
            { id: "meta-learning", group: "advanced", level: 3 },
            
            // Research nodes
            { id: "program-synthesis", group: "agi", level: 4 },
            { id: "neuro-symbolic", group: "agi", level: 4 },
            { id: "ai-alignment", group: "agi", level: 4 }
        ],
        links: [
            // Prerequisites relationships
            { source: "linear-algebra", target: "ml-fundamentals" },
            { source: "probability", target: "ml-fundamentals" },
            { source: "calculus", target: "neural-networks" },
            { source: "ml-fundamentals", target: "neural-networks" },
            { source: "python-basics", target: "scientific-python" },
            { source: "neural-networks", target: "transformers" },
            { source: "neural-networks", target: "reinforcement-learning" },
            { source: "algorithms", target: "program-synthesis" },
            { source: "transformers", target: "program-synthesis" },
            { source: "neural-networks", target: "neuro-symbolic" },
            { source: "algorithms", target: "neuro-symbolic" }
        ]
    }
};

// Progress tracking data structure
const progressData = {
    completedTopics: [],
    currentFocus: ["linear-algebra", "python-basics", "ml-fundamentals"],
    skillLevels: {
        "Mathematics": 30,
        "Programming": 40,
        "Machine Learning": 20,
        "Deep Learning": 10,
        "Computer Vision": 5,
        "NLP": 15,
        "Reinforcement Learning": 0,
        "Research Skills": 5
    },
    projectsCompleted: [],
    timeSpent: {
        total: 0,
        byTopic: {}
    },
    achievements: [
        "🎯 Started AI Journey",
        "📚 Roadmap Created"
    ]
};