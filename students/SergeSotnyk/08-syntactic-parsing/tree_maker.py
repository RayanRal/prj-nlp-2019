from random import Random
from typing import List, Dict

from conllu import TokenList
from sklearn.pipeline import Pipeline

from conllu_utils import TEST_FILENAME, load_trees, relation_to_features
from weak_classifier import load_weak_model, extract_x_y

_rnd = Random(1974)


def random_genome_from_tokens(tokens_num: int) -> List[int]:
    max_head_num = tokens_num
    return [_rnd.randint(0, max_head_num) for _ in range(tokens_num)]


def genome_from_sentence_tree(tree: List[TokenList]) -> List[int]:
    result = []
    for t in tree:
        head = t['head']
        if head is None:  # Some trees are not well formed
            return
        result.append(head)
    return result


def tokens_from_sentence_tree(tree: List[TokenList]) -> List[str]:
    return [t["form"] for t in tree]


def _calc_penalties(genome: List[int]) -> float:
    def has_cycle(g: int) -> bool:
        current = g
        for _ in range(len(genome)):
            if current == 0:
                return False
            current = genome[current - 1]
        return True

    result = 0.0
    # root should be one only
    zeros = sum(1 for g in genome if g == 0)
    result += 10. * abs(zeros - 1)
    # all ways should be finished in the root
    for g in genome:
        if has_cycle(g):
            result += 10.
    return result


def calc_genome_fitness(genome: List[int], tokens: List[str], classifier: Pipeline, cache) -> float:
    fitness = 0
    for i, head_pos in enumerate(genome):
        child_pos = i + 1
        rel = (child_pos, head_pos)
        if rel in cache:
            fit = cache[rel]
        else:
            f = relation_to_features(tokens, child_pos, head_pos)
            prediction = classifier.predict_proba([f])
            fit = prediction[0][1] - prediction[0][0]
            cache[rel] = fit
        fitness += fit
    fitness -= _calc_penalties(genome)
    return fitness


def generate_new_generation(sorted_genomes, mutation_prob, elitism: bool = False) -> List[List[int]]:
    result = []
    if elitism:
        result.append(sorted_genomes[0][1])
    pool_size = len(sorted_genomes)
    genome_size = len(sorted_genomes[0][1])
    max_rnd = pool_size - 1
    for i in range(pool_size):
        if len(result) >= pool_size:
            break
        # tournament to be a parent
        a, b, c, d = [_rnd.randint(0, max_rnd) for _ in range(4)]
        a = min(a, b)
        b = min(c, d)
        # select crossover points
        p1, p2 = 0, 0
        while p1 == p2:
            p1, p2 = _rnd.randint(0, genome_size - 1), _rnd.randint(0, genome_size - 1)
        if p1 > p2:
            p1, p2 = p2, p1
        ga = sorted_genomes[a][1]
        gb = sorted_genomes[b][1]
        child = ga[:p1] + gb[p1:p2] + ga[p2:]
        # mutation
        if _rnd.random() < mutation_prob:
            child[_rnd.randint(0, genome_size - 1)] = _rnd.randint(0, genome_size)
        result.append(child)
    return result


def restore_genome(tokens: List[str], classifier: Pipeline, verbose: bool = True) -> List[int]:
    pool_size = 1000
    max_epoch = 1000
    cross_prob = 0.3
    mutation_prob = 0.1
    pool = [random_genome_from_tokens(len(tokens)) for _ in range(pool_size)]
    if verbose:
        print("Start evolution")
    cache = {}
    prev_best_fitness = -1234567.0
    epochs_wo_changes = 0
    for epoch in range(max_epoch):
        fitnesses = [calc_genome_fitness(genome, tokens, classifier, cache) for genome in pool]
        sorted_genomes = sorted(zip(fitnesses, pool), reverse=True)
        if sorted_genomes[0][0] == prev_best_fitness:
            epochs_wo_changes += 1
            if epochs_wo_changes > 5:
                if verbose:
                    print("Early evolution stop at epoch", epoch)
                break
        else:
            epochs_wo_changes = 0
            prev_best_fitness = sorted_genomes[0][0]
        if verbose:
            print(f"Epoch {epoch}, the best genome: ({sorted_genomes[0][0]}){sorted_genomes[0][1]}")
        pool = generate_new_generation(sorted_genomes, mutation_prob, elitism=False)
    return pool[0]


def main():
    classifier = load_weak_model()
    test_trees = load_trees(TEST_FILENAME)
    print(f"{len(test_trees)} sentences loaded.")
    tree = test_trees[0]
    tokens = tokens_from_sentence_tree(tree)
    print("Tokens:", tokens)
    genome = genome_from_sentence_tree(tree)
    print("True genome  :", genome)
    rand_genome = random_genome_from_tokens(len(tree))
    print("Random genome:", rand_genome)

    f_genome = calc_genome_fitness(genome, tokens, classifier, {})
    f_rand_genome = calc_genome_fitness(rand_genome, tokens, classifier, {})

    print("Genome fitness = ", f_genome)
    print("Random genome fitness = ", f_rand_genome)

    restored_genome = restore_genome(tokens, classifier)
    print("Restored: ", restored_genome)

    for i in range(1, 10):
        tree = test_trees[i]
        tokens = tokens_from_sentence_tree(tree)
        print("Tokens:", tokens)
        genome = genome_from_sentence_tree(tree)
        f_genome = calc_genome_fitness(genome, tokens, classifier, {})
        print(f"True genome: ({f_genome}){genome}")
        restored_genome = restore_genome(tokens, classifier, verbose=False)
        f_restored_genome = calc_genome_fitness(restored_genome, tokens, classifier, {})
        print(f"Restored   : ({f_restored_genome}){restored_genome}")
        print()


if __name__ == "__main__":
    main()
