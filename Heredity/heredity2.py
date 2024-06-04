import csv
import itertools
import sys

import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": { 
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: { 
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}

UNCOND_PROBS = {
    True: PROBS['gene'][0] * PROBS['trait'][0][True] + PROBS['gene'][1] * PROBS['trait'][1][True] + PROBS['gene'][2] * PROBS['trait'][2][True],
    False: PROBS['gene'][0] * PROBS['trait'][0][False] + PROBS['gene'][1] * PROBS['trait'][1][False] + PROBS['gene'][2] * PROBS['trait'][2][False]
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def parent_probability(is_trait, genes):
    uncond_prob = PROBS['gene'][genes]
    if is_trait is None:
        return uncond_prob
    elif is_trait:
        return PROBS['trait'][genes][True] * uncond_prob / UNCOND_PROBS[True]
    else:
        return PROBS['trait'][genes][False] * uncond_prob / UNCOND_PROBS[False]


def child_probability(is_trait, genes):
    if genes == 0:
        uncond_prob = 1 #!!!!
    elif genes == 1:
        uncond_prob = 0.5
    else:
        uncond_prob = PROBS['gene'][0] * PROBS['mutation'] + PROBS['gene'][2] * (1 - PROBS['mutation'])
    if is_trait is None:
        return uncond_prob
    elif is_trait:
        return PROBS['trait'][genes][True] * uncond_prob / UNCOND_PROBS[True]
    else:
        return PROBS['trait'][genes][False] * uncond_prob / UNCOND_PROBS[False]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1
    children = {}
    parents = {}
    parents_genes = []
    for person in people:
        joint_prob[person] = []
        if people[person]['mother']:
            children[person] = people[person]
        else:
            parents[person] = people[person]

    for person in parents:
        person_trait = people[person]['trait']
        if person in one_gene:
            prob = parent_probability(person_trait, 1)
            parents_genes.append(1)
        elif person in two_genes:
            prob = parent_probability(person_trait, 2)
            parents_genes.append(2)
        else:
            prob = parent_probability(person_trait, 0)
            parents_genes.append(0)

        joint_prob *= prob
    
    for person in children:
        person_trait = people[person]['trait']
        if person in one_gene:
            prob = child_probability(person_trait, 1, parents_genes)   
        elif person in two_genes:
            prob = child_probability(person_trait, 2, parents_genes)
        else:
            prob = child_probability(person_trait, 0, parents_genes)

        joint_prob *= prob
    
    for person in people:
        if person in have_trait:
            trait_prob = PROBS['gene'][0] * PROBS['trait'][0][True] + PROBS['gene'][1] * PROBS['trait'][1][True] + PROBS['gene'][2] * PROBS['trait'][2][True]
            joint_prob[person].append(trait_prob)
        else:
            trait_prob = PROBS['gene'][0] * PROBS['trait'][0][False] + PROBS['gene'][1] * PROBS['trait'][1][False] + PROBS['gene'][2] * PROBS['trait'][2][False]
            joint_prob[person].append(trait_prob)
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in p:
        if name in one_gene:
            probabilities[name]['gene'][1] = p[name][0]
        elif name in two_genes:
            probabilities[name]['gene'][2] = p[name][0]
        else:
            probabilities[name]['gene'][0] = p[name][0]
        if name in have_trait:
            probabilities[name]['trait'][True] = p[name][1]
        else:
            probabilities[name]['trait'][False] = p[name][1]


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for prob in probabilities:
        person = probabilities[prob]
        gene_prob_alpha = person['gene'][0] + person['gene'][1] + person['gene'][2]
        person['gene'][0] = person['gene'][0] / gene_prob_alpha
        person['gene'][1] = person['gene'][1] / gene_prob_alpha
        person['gene'][2] = person['gene'][2] / gene_prob_alpha
        trait_prob_alpha = person['trait'][True] + person['trait'][False]
        person['trait'][True] = person['trait'][True] / trait_prob_alpha
        person['trait'][False] = person['trait'][False] / trait_prob_alpha


if __name__ == "__main__":
    main()
