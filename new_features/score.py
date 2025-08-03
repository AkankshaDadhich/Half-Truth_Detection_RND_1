import pandas as pd
import openai
import os

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_deceptive_statements(text):
    prompt = f"""
Original Paragraph:
{text}

Available Techniques::
Paraphrasing: Alter wording slightly while maintaining the core message, adding subtle misrepresentation.
Text Perturbations: Change specific words (e.g., adding negation or using synonyms) to shift meaning.
Adversarial Attacks: Modify key phrases or sentiment to produce misleading versions.
Omission: Exclude critical information that affects context.
Exaggeration: Overstate aspects to intensify the situation.
Understatement: Downplay important elements.
Alteration of Facts: Modify specific details, such as dates, figures, or entities.
Over-Representation of Numbers: Inflate or distort numerical data.
Generalization: Broaden details to obscure specifics.
Context Manipulation: Alter context to mislead.
Ambiguity: Use vague or unclear language.
Quantifier Shift: Change quantifiers (e.g., "some" to "most") to misrepresent magnitude.
Selective Comparison: Emphasize favorable details for bias.
False Equivalence: Compare unrelated events to imply misleading connections.
Misleading Cause and Effect: Imply causation without evidence.
Emotional Appeal: Use charged language to evoke emotions, distracting from facts.

Each statement should follow this template structure to ensure no empty strings:
Statement:"Generated deceptive statement here"
Rating:Rating here (True, Mostly True, Half-True, Mostly False, or False)"
Technique:"Primary technique(s) used here"

Format Example:
Deceptive Statement: [Generated Statement] Rating: Mostly False Technique: Omission (does not mention [critical omitted context])

"""
    
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',  # Change model to 'gpt-4' if available; fallback to 'gpt-3.5-turbo' if not
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Splitting response into statements and associated techniques and ratings
    deceptive_statements = response['choices'][0]['message']['content'].strip().split('\n')
    print(deceptive_statements)
    # deceptive_statements = ['1. Statement: "George Papadopoulos revealed to an Australian diplomat in 2016 that Russia possessed compromising information about Hillary Clinton, which directly initiated the FBI\'s investigation into Trump\'s campaign."  ', 'Rating: Mostly True  ', 'Technique: Paraphrasing (slightly alters wording while maintaining core message, but implies a more direct causation than is explicit in the original)', '', '2. Statement: "According to the New York Times, Papadopoulos\' role in the Trump campaign was crucial, claiming he orchestrated significant meetings and decisions that influenced foreign policy."  ', 'Rating: Half-True  ', 'Technique: Understatement (downplays the limited role described in the original and exaggerates his influence)', '', '3. Statement: "After meeting with Papadopoulos, the Australian government decided to contact the FBI because they felt Trump was colluding with Russia."  ', 'Rating: Mostly False  ', "Technique: Misleading Cause and Effect (implies a direct conclusion made by the Australian government that wasn't stated in the original)", '', '4. Statement: "Trump\'s White House insists that Papadopoulos was involved in the campaign\'s major strategic decisions and that his criminal charges reflect a broad conspiracy involving top campaign officials."  ', 'Rating: False  ', "Technique: Exaggeration (inflates the importance of Papadopoulos' role and the implications of his charges)", '', '5. Statement: "Mueller\'s investigation has primarily focused on the actions of Papadopoulos, leading many to question the integrity of the entire Trump campaign."  ', 'Rating: Mostly False  ', 'Technique: Omission (ignores the investigation also involves multiple other associates and the broader context of accusations against Trump)']
    # deceptive_statements = ['1. Deceptive Statement: "Lindsey Graham openly supports shutting down the Russia investigation despite its importance to the Trump administration."  ', '   Rating: Mostly False  ', '   Technique: Omission (fails to mention his support for allowing the investigation to continue)', '', '2. Deceptive Statement: "Most Republican lawmakers are now unified in their call to end Robert Mueller\'s inquiry into Russia\'s involvement in the election."  ', '   Rating: Mostly False  ', '   Technique: Generalization (implies a widespread consensus among Republicans, which is not accurate)', '', '3. Deceptive Statement: "Trump stated that he has complete faith in Mueller\'s investigation and believes it will clear his name."  ', '   Rating: Half-True  ', "   Technique: Text Perturbations (misinterprets Trump's comments, which were more measured than implied)", '', '4. Deceptive Statement: "Graham acknowledged that the investigation will likely find wrongdoing among Trump\'s allies, given the evidence collected so far."  ', '   Rating: Mostly True  ', '   Technique: Misleading Cause and Effect (implies Graham expects findings of wrongdoing, although he has not explicitly claimed this)', '', '5. Deceptive Statement: "The Russia investigation has consistently found serious crimes among numerous members of the Trump campaign and administration."  ', '   Rating: False  ', "   Technique: Exaggeration (overstates the findings of serious crimes, leading to a misleading interpretation of the investigation's outcomes)"]
    statements_data = []
    for i in range(0, len(deceptive_statements), 4):  # Step by 4 to account for empty string lines
        if i + 2 < len(deceptive_statements):  # Ensure we have enough items
            statement = deceptive_statements[i].replace('Statement:', '').strip().strip('"')
            rating = deceptive_statements[i + 1].replace('Rating:', '').strip()
            technique = deceptive_statements[i + 2].replace('Technique:', '').strip()
            statements_data.append((statement, text, rating, technique))

    return statements_data



def main():
    # Load the CSV file
    csv_file = '/home/akanksha-dadhich/Desktop/nlp rnd/new_features/20_true.csv'  # Change to your CSV file path
    data = pd.read_csv(csv_file)
    
    # Initialize a list to store results
    results = []
    statement_id = 1  # Start ID from 1
    
    # Iterate over each row in the CSV file
    for index, row in data.iterrows():
        title = row['title']
        text = row['text']
        subject = row['subject']
        date = row['date']
        
        # Generate multiple deceptive statements
        deceptive_statements = generate_deceptive_statements(text)
        
        # Append the statements to results
        for statement, evidence, rating, technique in deceptive_statements:
            results.append({
                "claim": statement,
                "evidence": evidence,
                "rating": rating,
                "techniques": technique
            })

    # Create a DataFrame for the results and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('giveforscoringnew_features_evaluation_results_testdata.csv', index=False, mode='a', header=False)

if __name__ == "__main__":
    main()

