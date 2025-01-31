import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from docx import Document
import google.generativeai as genai
import pandas as pd
import json
import os
from dotenv import load_dotenv
import io
import base64
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)

def extract_structured_data_with_gemini(doc_text):
    """Use Gemini to extract structured data from document text"""
    prompt = """
    Extract and categorize event data from the given list of competitions and participant numbers.
    Focus on the list that starts with "S. No" and contains "Name of the Competitions" and "Number of Participant".

    Categorize the events as follows:
    - Cultural: Doodle Art, Footloose, Nataki, Poetry, Salsa Workshop, Independence Day, Republic Day, Freshers Party
    - Technical: BGMI Squad, Stock Market Workshop, Tech events, Workshops
    - Sports: All sports activities
    - Other: Treasure Hunt and remaining events

    Calculate participants exactly as follows:
    - For "squads": multiply by 4 (e.g., "23 squads" = 92)
    - For "groups" or "teams": multiply by 4 (e.g., "8 groups" = 48)
    - For direct numbers: use as is (e.g., "3500" = 3500)
    - For workshop numbers: use exact number (e.g., "60" = 60)
    - When a '+' is present: use the base number (e.g., "70+" = 70)

    Return a clean JSON with calculated totals:
    {
        "events": [
            {
                "year": "2023-24",
                "category": "Cultural",
                "num_events": 0,
                "participants": 0
            }
        ]
    }

    Important:
    - Only process entries that have clear participant numbers
    - Group by category and sum participants
    - Return only valid JSON format
    - Do not include mathematical expressions in the output
    - Calculate all numbers before including in JSON
    """
    
    response = model.generate_content(prompt + doc_text)
    try:
        # Clean up the response text
        response_text = response.text
        print("Raw response:", response_text)  # Debug print
        
        # Extract only the JSON part
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            # Remove any markdown formatting
            json_str = json_str.strip('`').strip('json')
            data = json.loads(json_str)
            print("Parsed data:", json.dumps(data, indent=2))  # Debug print
            return data
        else:
            print("No valid JSON found in response")
            return None
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return None

def extract_data_from_unstructured_doc(docx_file):
    # Read the document
    doc = Document(docx_file)
    
    # Extract text from tables focusing on the relevant section
    relevant_data = []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) >= 3:
                # Check if this is the header or contains relevant data
                if any(x in cells[0].lower() for x in ['s.', 'no', 'name']) or \
                   any(x in cells[1].lower() for x in ['competition', 'event']):
                    relevant_data.append("\t".join(cells))
                # Check for participant numbers
                elif any(x in str(cells[-2]).lower() for x in ['squad', 'group', 'team']) or \
                     any(char.isdigit() for char in str(cells[-2])):
                    relevant_data.append("\t".join(cells))
    
    # Format table data
    table_text = "List of Competitions and Participants:\n"
    table_text += "\n".join(relevant_data)
    print("Extracted table text:", table_text)  # Debug print
    
    # Extract structured data using Gemini
    structured_data = extract_structured_data_with_gemini(table_text)
    
    if not structured_data or 'events' not in structured_data:
        print("Failed to extract structured data")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(structured_data['events'])
    
    # Rename columns
    column_mapping = {
        'year': 'Year',
        'category': 'Category',
        'num_events': 'Events',
        'participants': 'Participants'
    }
    df = df.rename(columns=column_mapping)
    
    print("Final DataFrame:", df)  # Debug print
    return df

def create_plot(plot_type, data, selected_year=None, selected_categories=None):
    # Clear any existing plots
    plt.close('all')
    
    # Set figure DPI and style
    plt.rcParams['figure.dpi'] = 100
    plt.style.use('seaborn')
    plt.rcParams['font.size'] = 10
    
    # Filter data
    filtered_data = data[data["Category"].isin(selected_categories)]
    
    # Create figure and axes before plotting
    if plot_type == "trends":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create vertical bar plot
        bars = ax.bar(range(len(filtered_data)), 
                     filtered_data["Events"],
                     color=plt.cm.Pastel1(np.linspace(0, 1, len(filtered_data))))
        
        # Customize the plot
        ax.set_xticks(range(len(filtered_data)))
        ax.set_xticklabels(filtered_data["Category"], rotation=45, ha='right')
        ax.set_ylabel("Number of Events")
        ax.set_title("Events Distribution by Category", pad=20)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height)}',
                   ha='center', va='bottom')
    
    elif plot_type == "bar":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[1, 1])
        
        # Events subplot
        bars1 = ax1.bar(filtered_data["Category"], 
                       filtered_data["Events"],
                       color=plt.cm.Pastel1(np.linspace(0, 1, len(filtered_data))))
        ax1.set_xticklabels(filtered_data["Category"], rotation=45, ha='right')
        ax1.set_ylabel("Number of Events")
        ax1.set_title("Events by Category")
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Participants subplot
        if 'Participants' in filtered_data.columns and not filtered_data['Participants'].isna().all():
            bars2 = ax2.bar(filtered_data["Category"], 
                          filtered_data["Participants"],
                          color=plt.cm.Pastel2(np.linspace(0, 1, len(filtered_data))))
            ax2.set_xticklabels(filtered_data["Category"], rotation=45, ha='right')
            ax2.set_ylabel("Number of Participants")
            ax2.set_title("Participants by Category")
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height):,}',
                        ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Participants Data Available',
                    ha='center', va='center',
                    transform=ax2.transAxes)
    
    elif plot_type == "pie":
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if 'Participants' in filtered_data.columns and not filtered_data['Participants'].isna().all():
            # Calculate percentages
            total = filtered_data['Participants'].sum()
            
            # Create pie chart
            ax.pie(filtered_data['Participants'],
                   labels=filtered_data['Category'],
                   autopct=lambda pct: f'{pct:.1f}%\n({int(total * pct/100):,})',
                   colors=plt.cm.Pastel1(np.linspace(0, 1, len(filtered_data))),
                   startangle=90)
            
            ax.set_title('Participant Distribution by Category')
        else:
            ax.text(0.5, 0.5, 'No Participants Data Available',
                    ha='center', va='center',
                    transform=ax.transAxes)
    
    elif plot_type == "radar":
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        categories = filtered_data["Category"]
        events = filtered_data["Events"]
        participants = filtered_data["Participants"]
        
        # Normalize the values
        events_norm = events / events.max()
        participants_norm = participants / participants.max()
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        
        # Close the plot by appending first values
        events_norm = np.concatenate((events_norm, [events_norm[0]]))
        participants_norm = np.concatenate((participants_norm, [participants_norm[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, events_norm, 'o-', label='Events')
        ax.plot(angles, participants_norm, 'o-', label='Participants')
        ax.fill(angles, events_norm, alpha=0.25)
        ax.fill(angles, participants_norm, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Category Performance Analysis')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    elif plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a matrix of normalized values
        matrix = np.array([
            filtered_data["Events"] / filtered_data["Events"].max(),
            filtered_data["Participants"] / filtered_data["Participants"].max()
        ])
        
        sns.heatmap(matrix, 
                    annot=True, 
                    fmt='.2f',
                    xticklabels=filtered_data["Category"],
                    yticklabels=['Events', 'Participants'],
                    cmap='YlOrRd',
                    ax=ax)
        
        ax.set_title('Participation Density Analysis')
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Please upload a Word document'}), 400
    
    try:
        df = extract_data_from_unstructured_doc(file)
        if df is None:
            return jsonify({'error': 'Could not extract data from document'}), 400
        
        data = {
            'data': df.to_dict('records'),
            'years': sorted(df['Year'].unique().tolist()),
            'categories': df['Category'].unique().tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    data = request.json
    df = pd.DataFrame(data['data'])
    plot_type = data['plot_type']
    selected_year = data.get('selected_year')
    selected_categories = data.get('selected_categories', [])
    
    plot_url = create_plot(plot_type, df, selected_year, selected_categories)
    return jsonify({'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)
