data = [ 
{"Color": "Red", "Type": "Sports", "Origin": "Domestic", "Stolen": "Yes"}, 
{"Color": "Red", "Type": "Sports", "Origin": "Domestic", "Stolen": "No"}, 
{"Color": "Red", "Type": "Sports", "Origin": "Domestic", "Stolen": "No"}, 
{"Color": "Yellow", "Type": "Sports", "Origin": "Domestic", "Stolen": "No"}, 
{"Color": "Yellow", "Type": "Sports", "Origin": "Imported", "Stolen": "Yes"}, 
{"Color": "Yellow", "Type": "SUV", "Origin": "Imported", "Stolen": "Yes"}, 
{"Color": "Yellow", "Type": "SUV", "Origin": "Imported", "Stolen": "Yes"}, 
{"Color": "Yellow", "Type": "SUV", "Origin": "Domestic", "Stolen": "No"}, 
{"Color": "Red", "Type": "SUV", "Origin": "Domestic", "Stolen": "No"}, 
{"Color": "Red", "Type": "Sports", "Origin": "Imported", "Stolen": "Yes"}, 
] 

def conditional_prob(attribute, value, label, dataset): 
    count_label = sum(1 for row in dataset if row["Stolen"] == label) 
    count_attr_given_label = sum(1 for row in dataset if row["Stolen"] == label and 
    row[attribute] == value) 
    return count_attr_given_label / count_label if count_label else 0 

def predict(x, dataset): 
    total = len(dataset) 
    
    yes_count = sum(1 for row in dataset if row["Stolen"] == "Yes") 
    no_count = total - yes_count 
    P_yes = yes_count / total
    P_no = no_count / total
    
    P_x_given_yes = ( 
        conditional_prob("Color", x["Color"], "Yes", dataset) * conditional_prob("Type", x["Type"], "Yes", dataset) * conditional_prob("Origin", x["Origin"], "Yes", dataset) 
    ) 
    
    P_x_given_no = ( 
        conditional_prob("Color", x["Color"], "No", dataset) * conditional_prob("Type", x["Type"], "No", dataset) * conditional_prob("Origin", x["Origin"], "No", dataset) 
    ) 
    
    posterior_yes = P_x_given_yes * P_yes 
    posterior_no = P_x_given_no * P_no 
    
    prediction = "Yes" if posterior_yes > posterior_no else "No" 
    
    return prediction, posterior_yes, posterior_no 

test_input = {"Color": "Yellow", "Type": "Sports", "Origin": "Domestic"} 
prediction, py, pn = predict(test_input, data) 

print("--- Naive Bayes Prediction for New Input ---") 
print(f"Test Input: {test_input}") 
print(f"P(x|Yes) * P(Yes) [Stolen]: {py:.3f}") 
print(f"P(x|No) * P(No) [Not Stolen]: {pn:.3f}") 
print(f"Prediction: {prediction}")