from nim import train, play

best_performance = 0
best_hyperparameters = {'epsilon': None, 'alpha': None}

for epsilon in [0.1, 0.2, 0.3]:
    for alpha in [0.1, 0.2, 0.3]:
        nim_ai = NimAI(alpha=alpha, epsilon=epsilon)
        trained_ai = train(10000)  # Train for a certain number of iterations
        # Evaluate the performance of the trained AI (you need to define a performance metric)
        performance = evaluate_performance(trained_ai)
        
        # Update best hyperparameters if needed
        if performance > best_performance:
            best_performance = performance
            best_hyperparameters['epsilon'] = epsilon
            best_hyperparameters['alpha'] = alpha

print("Best Hyperparameters:", best_hyperparameters)