import argparse
import csv
import sys

from util import Node, QueueFrontier, StackFrontier

# Maps names to a set of corresponding person_ids
people_to_ids = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory, print_messages):
    """
    Load data from CSV files into memory
    """

    if print_messages:
        print(f"Loading data from '{directory}' ...")

    # Load people and construct people_to_ids mapping
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in people_to_ids:
                people_to_ids[row["name"].lower()] = {row["id"]}
            else:
                people_to_ids[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars and link them to people and movies
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass

    if print_messages:
        print("Data loaded.")


def main(directory, is_verbose=True):

    # Load data from files into memory
    load_data(directory, is_verbose)

    name_prompt = "Name: " if is_verbose else ""
    continue_prompt = "Try again (Y/N)? " if is_verbose else ""

    while True:

        # Ask user for names
        star_name_1 = input(name_prompt)
        star_name_2 = input(name_prompt)

        # Find ID for the first star
        source = person_id_for_name(star_name_1, is_verbose)
        if source is None:
            sys.exit("Person 1 not found.")

        # Find ID for the second star
        target = person_id_for_name(star_name_2, is_verbose)
        if target is None:
            sys.exit("Person 2 not found.")

        # Find shortest path between source and target stars
        path = shortest_path(source, target)

        # print names (in quiet mode)
        if not is_verbose:
            print(f"{star_name_1} (ID: {source}) - {star_name_2} (ID: {target})")

        if path is None:
            print("Not connected.")
        else:
            degrees = len(path)
            print(f"{degrees} degrees of separation.")
            path = [(None, source)] + path
            if is_verbose:
                for i in range(degrees):
                    person1 = people[path[i][1]]["name"]
                    person2 = people[path[i + 1][1]]["name"]
                    movie = movies[path[i + 1][0]]["title"]
                    print(f"{i + 1}: {person1} and {person2} starred in {movie}")

        # print empty line at the end (in quiet mode)
        if not is_verbose:
            print("\n", end="")

        give_another_pair = input(continue_prompt)
        if (give_another_pair.upper() != "Y"):
            break

def construct_path(src, trg):
    # TODO: 
    this_path = []
    node = trg
    while node != src:
        this_path.insert(0, (node.action, node.state))
        node = node.parent
    return this_path

def make_full_path(src_node, trg_node, between_node):
    path_from_source = construct_path(src_node, between_node)
    path_from_target = construct_path(trg_node, between_node)[1:]

    return path_from_source + path_from_target[::-1]

def shortest_path(source, target):
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    source_node = Node(state=source, parent=None, action=None)
    target_node = Node(state=target, parent=None, action=None)
    src_explored = set()
    trg_explored = set()
    src_frontier = QueueFrontier()
    trg_frontier = QueueFrontier()
    src_frontier.add(source_node)
    trg_frontier.add(target_node)
    while not src_frontier.empty() and not trg_frontier.empty():
        src_node = src_frontier.remove()
        src_explored.add(src_node)
        neighbors = neighbors_for_person(src_node.state)
        for movie, actor in neighbors:
            if actor not in src_explored and not src_frontier.contains_state(actor):
                new_node = Node(state=actor, parent=src_node, action=movie)
                if new_node.state in trg_explored:
                    print("found from src")
                    return make_full_path(source_node, target_node, new_node)
                else:
                    src_frontier.add(new_node)

        trg_node = trg_frontier.remove()
        trg_explored.add(trg_node)
        neighbors = neighbors_for_person(trg_node.state)
        for movie, actor in neighbors:
            if actor not in trg_explored and not trg_frontier.contains_state(actor):
                new_node = Node(state=actor, parent=trg_node, action=movie)
                if new_node.state in src_explored:
                    print("found from trg")
                    return make_full_path(source_node, target_node, new_node)
                else:
                    trg_frontier.add(new_node)
    
    
    return None
    


def person_id_for_name(name, is_verbose):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(people_to_ids.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        if is_verbose:
            print(f"Which '{name}'?")

            for person_id in person_ids:
                person = people[person_id]
                name = person["name"]
                birth = person["birth"]
                print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            id_prompt = "Intended Person ID: " if is_verbose else ""
            person_id = input(id_prompt)
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """

    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))

    return neighbors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find degrees of separation between two actors")
    parser.add_argument(
        'directory',
        help='load the data files from this directory, defaults to "large"',
        nargs="?",
        default="large")
    parser.add_argument(
        "-q", "--quiet",
        help="disable user prompts and other extra output (used by the grader)",
        action="store_true")

    args = parser.parse_args()
    verbose = not args.quiet
    main(args.directory, verbose)
