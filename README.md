# ☁️ **Cloud at AI Sky: Optimizing Cloud Resource Management**

![Banner](https://github.com/user-attachments/assets/f54b05af-137a-46dd-9fde-4d4ec9d57bbb)  

---

## 📌 **Project Overview**

**Cloud at AI Sky** is a research-driven project that optimizes resource management in cloud computing environments. Inspired by [Feldman et al. (2011)](https://www.researchgate.net/publication/236964785_Using_Approximate_Dynamic_Programming_to_Optimize_Admission_Control_in_Cloud_Computing_Environment), it combines:

- **MDP Modeling** for dynamic resource allocation  
- **Dynamic Programming-Driven Admission Control** for revenue optimization  
- **Load Balancing Algorithms** for efficient resource utilization  

---

## 🚀 **Key Features**

### 1. Markov Decision Process (MDP) Modeling
- Captures cloud infrastructure states (VM deployments, pending requests)  
- Defines actions: admit/reject requests based on resource constraints  

### 2. Bellman Equation-Driven Admission Control
- Prioritizes high-revenue VM deployments  
- Maximizes long-term revenue while respecting resource limits  

### 3. Hybrid Load Balancing
- **Greedy Algorithm:** Allocates to most underutilized nodes  
- **Randomized Algorithm:** Prevents node overload through stochastic distribution  

---

## 🧬 **Methodology**

### 1. Continuous-Time MDP Framework
- Models cloud environment as CTMDP  
- States track active deployments + pending requests  
- Transition probabilities account for arrivals/departures  

### 2. Dynamic Programming Optimization
- Solves Bellman equation for value functions  
- Derives optimal admission policies  

### 3. Resource Allocation Strategies
- Capacity-aware placement decisions  
- Real-time load balancing
  
---

## 🛠️ **Technical Implementation**

**Core Stack**  
- Python (Simulation Engine)  
- NumPy (Numerical Computations)  
- SciPy (Optimization Tasks)  

**Key Components**  
- State-space enumeration with capacity constraints  
- Policy iteration for MDP resolution  
- Load balancing decision modules  

---

## 🌟 **Unique Value Proposition**

1. **Research-Backed Architecture**  
   Implements peer-reviewed CTMDP framework from Feldman et al.  

2. **Dual-Mode Operation**  
   Supports both greedy optimization and randomized fairness  

---

## 📬 **Connect & Collaborate**

[📧 Email](mailto:s.shivnaran@gmail.com) | [📄 Full Paper](https://www.researchgate.net/publication/236964785_Using_Approximate_Dynamic_Programming_to_Optimize_Admission_Control_in_Cloud_Computing_Environment)  

**Open For:**
- 🔍 Research collaborations in cloud computing and AI.  
- 💼 Industry roles in cloud resource management and optimization.  

---
