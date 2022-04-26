import csv
import itertools
import sys

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
    output = 1

    # Calculate the probability of each person having the indicated number of genes and update the joint probability
    for person in set(people):
        # Initialize the parents' gene counts to None
        mother_gene_count = None
        father_gene_count = None
        # If the person's parents are identified, check the number of genes for the person's parents
        mother = people[person]["mother"]
        father = people[person]["father"]
        if mother is not None and father is not None:
            mother_gene_count = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_gene_count = 1 if father in one_gene else 2 if father in two_genes else 0

        # Calculate probability of the person having a given number of genes based on their parents' gene count and update our output
        # The gene count for our computation will depend on whether the person is in one_gene, two_genes or neither
        gene_count = 1 if person in one_gene else 2 if person in two_genes else 0
        # 'probability_of_genes' is our own helper function
        output *= probability_of_genes(mother_gene_count, father_gene_count, gene_count)

        # Given the person's gene count, we can then calculate the probability that they have the trait or not from our global table and update our output
        trait_value = person in have_trait
        output *= PROBS["trait"][gene_count][trait_value]
        
    return output
    
    # raise NotImplementedError


def probability_of_genes(mother_gene_count, father_gene_count, gene_count):
    """
    This is our own function that will calculate the probability that a given 'person' will have 'gene_count' number of genes
    The probability will depend on how may genes the person's parents have
    """
    # If no parental information, we just return the unconditional probability from our global table
    if mother_gene_count is None and father_gene_count is None:
        return PROBS["gene"][gene_count]
    else:
        # We know that from each parent, a child can either acquire 0 or 1 of the gene, thus we can build a probability distribution like so
        # Where the key = number of genes of one parent, the inner_key = number of genes acquired by the child from that parent,
        # and value = probability that child acquires inner_key # of genes from the parent with key # of genes
        # Thus, for example, prob_acquire_gene[0][1] is the probability that a child acquires 1 gene from a parent with 0 gene count
        prob_acquire_gene = {
            0: {
                # If the parent has 0 gene, then the child can only acquire the gene from the parent via mutation
                0: 1 - PROBS["mutation"],
                1: PROBS["mutation"]
            },
            1: {
                # If the parent has 1 gene, then the child can have 0 genes if they acquire the non-gene that doesn't mutate
                # Or they acquire the gene which then mutates; the opposite is true for the child to acquire 1 gene
                # Note that both cases evaluate to 0.5
                0: 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"],
                1: 0.5 * PROBS["mutation"] + 0.5 * (1 - PROBS["mutation"])
            },
            2: {
                # If the parent has 2 genes, then the child will definitely inherit the gene, it's now a matter of whether the gene mutates or not
                0: PROBS["mutation"],
                1: 1 - PROBS["mutation"]
            }
        }
        
        # Based on the above distribution, we can calculate the probability that a child acquires 0, 1 or 2 genes
        if gene_count == 0:
            # Child acquires 0 gene from both parents
            return prob_acquire_gene[mother_gene_count][0] * prob_acquire_gene[father_gene_count][0]
        elif gene_count == 2:
            # Child acquires 1 gene from each parent
            return prob_acquire_gene[mother_gene_count][1] * prob_acquire_gene[father_gene_count][1]
        else:
            # Child acquires 1 gene from one parent and 0 from the other
            return (prob_acquire_gene[mother_gene_count][0] * prob_acquire_gene[father_gene_count][1]
                    + prob_acquire_gene[mother_gene_count][1] * prob_acquire_gene[father_gene_count][0])


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Iterate through each person in probabilities
    for person in set(probabilities):
        # Person's gene_count in this scenario
        gene_count = 1 if person in one_gene else 2 if person in two_genes else 0
        # Person's trait value
        trait_value = person in have_trait
        # Update the probabilities value for the person
        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][trait_value] += p

    # raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Iterate through each person in probailities
    for person in set(probabilities):
        # Compute the multiplier for gene values
        gene_multiplier = 1 / (sum(probabilities[person]["gene"][gene_count] for gene_count in range(3)))
        # Update the values
        for gene_count in range(3):
            probabilities[person]["gene"][gene_count] *= gene_multiplier
        
        # Compute the multiplier for trait values
        trait_multiplier = 1 / sum(probabilities[person]["trait"][trait_value] for trait_value in [True, False])
        # Update the values
        for trait_value in [True, False]:
            probabilities[person]["trait"][trait_value] *= trait_multiplier
    
    # raise NotImplementedError


if __name__ == "__main__":
    main()
