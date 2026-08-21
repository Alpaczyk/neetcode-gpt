class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        for _ in range(iterations):
            x_old = init
            init = x_old - (learning_rate * 2 * x_old)
            print(init)
        
        return round(init, 5)
        

