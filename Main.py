import sys
import argparse
from GeneticAlgorithm import GeneticAlgorithm


#Main configurado para hacer la prueba con Irace
def main():
    parser = argparse.ArgumentParser()
    

    parser.add_argument("configurationID", type=int)
    parser.add_argument("instanceID", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("instance", type=str) 
    parser.add_argument("bound", type=str, nargs='?')  

    parser.add_argument("--population_size", type=int, required=True)
    parser.add_argument("--crossover_rate", type=float, required=True)
    parser.add_argument("--mutation_rate", type=float, required=True)
    parser.add_argument("--generations", type=int, required=True)
    
    args = parser.parse_args()
    
    try:
        ga = GeneticAlgorithm()
        best_accuracy = ga.algoritmo_genetico(
            numero_poblacion=args.population_size,
            p_cruza=args.crossover_rate,
            p_mutam=args.mutation_rate,
            n_generaciones=args.generations
        )
        

        sys.stdout.write(f"{best_accuracy:.6f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.stdout.write("0.0")  
        sys.exit(1)

if __name__ == "__main__":
    main()