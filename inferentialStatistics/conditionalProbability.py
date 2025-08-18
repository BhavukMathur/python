
def conditional_probability_of_support(selected_individuals):

    total_conservative_capitalists = 0
    supporters_of_environmentalism = 0

    for person in selected_individuals:
        if person.get('conservatism') and person.get('capitalism'):
            total_conservative_capitalists = total_conservative_capitalists + 1
            if person.get('environmentalism'):
                supporters_of_environmentalism = supporters_of_environmentalism + 1

    if total_conservative_capitalists == 0:
        return 0.0

    probability = supporters_of_environmentalism / total_conservative_capitalists
    return round(probability, 4)

# Input and output processing (do not edit)
from ast import literal_eval
print(round(conditional_probability_of_support(literal_eval(input())), 4))