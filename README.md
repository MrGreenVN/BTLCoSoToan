# Automatic Differentiation with PyTorch

## Overview

This project provides a comprehensive study of **Automatic Differentiation (AD)** and its role in modern Machine Learning and Deep Learning.

The notebook explains the mathematical foundations behind differentiation methods, computational graphs, chain rule propagation, backpropagation, and PyTorch's Autograd engine through both theoretical analysis and practical implementations.

## Contents

### 1. Introduction
- Motivation for gradient computation in Machine Learning
- Challenges of manual derivative calculation
- Importance of optimization and backpropagation

### 2. Differentiation Methods
Comparison among:

- Symbolic Differentiation
- Numerical Differentiation
- Automatic Differentiation

including:
- Mathematical formulation
- Advantages and disadvantages
- Python implementations using SymPy and PyTorch

### 3. Multivariable Chain Rule
- Function composition
- Computational graphs
- Multiple gradient paths
- Chain rule over complex networks

### 4. Backpropagation in PyTorch
- Gradient computation using Autograd
- Manual derivative verification
- Scalar and non-scalar outputs
- Comparison with symbolic and numerical approaches

### 5. Detaching Computation Graphs
- Understanding `detach()`
- Metadata vs Storage
- Gradient tracking mechanisms

### 6. Automatic Differentiation with Control Flow
- Loops
- Conditional statements
- Dynamic computational graphs
- Internal behavior of PyTorch Autograd

### 7. Exercises and Advanced Topics
- Higher-order derivatives
- Hessian computation
- Gradient accumulation
- Jacobian matrices
- Jacobian-Vector Products (JVP)
- Forward-mode vs Backward-mode differentiation
- Dependency graph construction
- Automatic differentiation of complex functions

## Technologies

- Python
- PyTorch
- SymPy
- NumPy
- Matplotlib
- Google Colab

## Learning Outcomes

After completing this project, readers will be able to:

- Understand the differences between symbolic, numerical, and automatic differentiation.
- Build and analyze computational graphs.
- Apply the chain rule to complex function compositions.
- Understand how backpropagation works in neural networks.
- Use PyTorch Autograd effectively.
- Compute gradients, Jacobians, Hessians, and higher-order derivatives.
- Analyze forward-mode and reverse-mode differentiation.

## Repository Structure
