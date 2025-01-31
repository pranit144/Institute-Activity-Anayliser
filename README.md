# Event Data Extraction and Visualization

This project is a Flask-based web application that extracts structured event data from unstructured Word documents (.docx), categorizes the events, and generates various visualizations based on the extracted data. It uses Google Gemini AI for data extraction and Matplotlib/Seaborn for data visualization.

## Features
- Extract event details (event name, category, participant count) from .docx files
- Categorize events into Cultural, Technical, Sports, and Other
- Generate structured JSON data
- Create visualizations including bar charts, pie charts, radar charts, and heatmaps
- Interactive web-based interface

## Technologies Used
- **Backend**: Python (Flask, Pandas, Matplotlib, Seaborn, NumPy, docx)
- **AI Processing**: Google Gemini AI for NLP-based data extraction
- **Frontend**: HTML, CSS, JavaScript (for web interface)
- **API**: REST API endpoints for data processing and visualization

## Installation & Setup

### Prerequisites
Make sure you have the following installed:
- Python 3.7+
- pip
- Virtual environment (optional but recommended)

### Steps
1. **Clone the repository**:
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up the environment variables**:
   - Create a `.env` file and add your Google Gemini API key:
   ```sh
   GOOGLE_API_KEY='your_api_key_here'
   ```

5. **Run the application**:
   ```sh
   python app.py
   ```

6. **Open your browser and navigate to**:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
1. Upload a Word document containing event details.
2. The system extracts and categorizes the event data.
3. Choose visualization options (bar chart, pie chart, heatmap, etc.).
4. Generate and view the visualized data.

## API Endpoints
### 1. Upload File
   - **Endpoint:** `/upload`
   - **Method:** `POST`
   - **Request:** Form-data with a `.docx` file
   - **Response:** Extracted event data in JSON format

### 2. Generate Plot
   - **Endpoint:** `/generate_plot`
   - **Method:** `POST`
   - **Request:** JSON containing extracted data and selected visualization type
   - **Response:** Base64-encoded image of the generated plot

## Supported Visualization Types
- **Bar Chart**: Shows the number of events and participants per category.
- **Pie Chart**: Displays the distribution of participants across event categories.
- **Radar Chart**: Compares different categories based on event count and participation.
- **Heatmap**: Highlights the density of participation across events.

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.

## License
This project is licensed under the MIT License.

## Contact
For any queries or contribution

