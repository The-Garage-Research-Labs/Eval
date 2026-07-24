import os
import random
import csv
from pathlib import Path
from flask import Flask, render_template_string, request, redirect, url_for, send_file

app = Flask(__name__)

# Base directory matching your path structure
BASE_DIR = Path("/home/abdo/PAPER/Eval/data/websrc/WebSRC_v1.0_test/release_testset")
CSV_PATH = Path("annotations.csv")

# HTML Template for the main interface
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Annotator</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 20px; background-color: #fcfcfc; color: #333; }
        .container { display: flex; flex-direction: column; align-items: center; max-width: 900px; margin: 0 auto; }
        .image-box { margin: 15px; border: 2px solid #ddd; border-radius: 8px; overflow: hidden; max-width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.05); background: white; }
        .image-box img { max-width: 100%; height: auto; max-height: 500px; display: block; }
        .buttons { margin: 15px; }
        button { font-size: 16px; padding: 12px 24px; margin: 5px; cursor: pointer; border-radius: 4px; font-weight: bold; border: none; transition: background-color 0.2s; }
        .btn-kv { background-color: #4CAF50; color: white; }
        .btn-kv:hover { background-color: #43a047; }
        .btn-comp { background-color: #2196F3; color: white; }
        .btn-comp:hover { background-color: #1e88e5; }
        .btn-table { background-color: #FF5722; color: white; }
        .btn-table:hover { background-color: #f4511e; }
        .btn-skip { background-color: #9e9e9e; color: white; }
        .btn-skip:hover { background-color: #757575; }
        .info { font-size: 15px; margin-bottom: 10px; line-height: 1.6; }
        table { width: 100%; max-width: 600px; margin-top: 30px; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        th, td { padding: 10px; border: 1px solid #eee; text-align: left; }
        th { background-color: #f5f5f5; }
        .delete-btn { background-color: #f44336; color: white; padding: 5px 10px; font-size: 12px; border: none; cursor: pointer; border-radius: 4px; margin: 0; }
        .delete-btn:hover { background-color: #d32f2f; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Web Annotation Tool</h2>
        <div class="info">
            <strong>Progress:</strong> {{ progress }} <br>
            <strong>Current Domain:</strong> {{ current_web }} <br>
            <strong>Showing file:</strong> {{ png_name }}
        </div>
        
        <div class="image-box">
            <img src="{{ url_for('get_image', path=png_rel_path) }}" alt="Website Screenshot">
        </div>
        
        <div class="buttons">
            <form action="/annotate" method="post" style="display: inline;">
                <input type="hidden" name="web_path" value="{{ current_web }}">
                <button type="submit" name="label" value="KV" id="btn-kv" class="btn-kv">[1] KV (Key-Value)</button>
                <button type="submit" name="label" value="Compare" id="btn-comp" class="btn-comp">[2] Compare</button>
                <button type="submit" name="label" value="Table" id="btn-table" class="btn-table">[3] Table</button>
                <button type="submit" name="label" value="Skip" id="btn-skip" class="btn-skip">[S] Skip</button>
            </form>
            <button type="button" onclick="location.reload();" class="btn-skip" style="background-color: #607d8b;">Show Different PNG</button>
        </div>

        {% if annotations %}
        <h3>Completed Annotations</h3>
        <table>
            <tr>
                <th>Website Path</th>
                <th>Label</th>
                <th>Action</th>
            </tr>
            {% for web, label in annotations.items() %}
            <tr>
                <td>{{ web }}</td>
                <td>{{ label }}</td>
                <td>
                    <form action="/delete" method="post" style="display:inline;">
                        <input type="hidden" name="web_path" value="{{ web }}">
                        <button type="submit" class="delete-btn">Delete</button>
                    </form>
                </td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>

    <script>
        // Set up keyboard shortcuts for speed
        document.addEventListener('keydown', function(event) {
            if (event.key === '1') {
                document.getElementById('btn-kv').click();
            } else if (event.key === '2') {
                document.getElementById('btn-comp').click();
            } else if (event.key === '3') {
                document.getElementById('btn-table').click();
            } else if (event.key.toLowerCase() === 's') {
                document.getElementById('btn-skip').click();
            }
        });
    </script>
</body>
</html>
"""

FINISHED_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Web Annotator - Completed</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 20px; background-color: #fcfcfc; }
        .container { max-width: 600px; margin: 0 auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
        th, td { padding: 10px; border: 1px solid #eee; text-align: left; }
        th { background-color: #f5f5f5; }
        .delete-btn { background-color: #f44336; color: white; padding: 5px 10px; font-size: 12px; border: none; cursor: pointer; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>All Web Annotations Completed!</h2>
        <p>Your choices are saved in <strong>annotations.csv</strong>.</p>
        
        <h3>Completed Annotations</h3>
        <table>
            <tr>
                <th>Website Path</th>
                <th>Label</th>
                <th>Action</th>
            </tr>
            {% for web, label in annotations.items() %}
            <tr>
                <td>{{ web }}</td>
                <td>{{ label }}</td>
                <td>
                    <form action="/delete" method="post" style="display:inline;">
                        <input type="hidden" name="web_path" value="{{ web }}">
                        <button type="submit" class="delete-btn">Delete</button>
                    </form>
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

def get_websites():
    websites = []
    # Reads the category folders (e.g. auto, game)
    for category_dir in BASE_DIR.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            # Reads the subdirectories (e.g. 02, 14)
            for web_dir in category_dir.iterdir():
                if web_dir.is_dir() and not web_dir.name.startswith('.'):
                    rel_path = web_dir.relative_to(BASE_DIR)
                    websites.append(str(rel_path))
    return sorted(websites)

def load_annotations():
    annotations = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    annotations[row[0]] = row[1]
    return annotations

def save_annotation(web_path, label):
    annotations = load_annotations()
    annotations[web_path] = label
    with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for k, v in sorted(annotations.items()):
            writer.writerow([k, v])

@app.route('/')
def index():
    websites = get_websites()
    annotations = load_annotations()
    
    remaining = [w for w in websites if w not in annotations]
    
    if not remaining:
        return render_template_string(FINISHED_TEMPLATE, annotations=annotations)
    
    current_web = remaining[0]
    web_dir = BASE_DIR / current_web
    png_files = list(web_dir.glob("**/*.png"))
    
    # Auto-handle empty directories to avoid blocking the workflow
    while remaining and not png_files:
        save_annotation(current_web, "No PNG Found")
        annotations = load_annotations()
        remaining = [w for w in websites if w not in annotations]
        if not remaining:
            return render_template_string(FINISHED_TEMPLATE, annotations=annotations)
        current_web = remaining[0]
        web_dir = BASE_DIR / current_web
        png_files = list(web_dir.glob("**/*.png"))

    random_png = random.choice(png_files)
    png_rel_path = random_png.relative_to(BASE_DIR)
    
    progress = f"{len(annotations)} / {len(websites)}"
    
    return render_template_string(
        INDEX_TEMPLATE,
        progress=progress,
        current_web=current_web,
        png_name=random_png.name,
        png_rel_path=str(png_rel_path),
        annotations=annotations
    )

@app.route('/image/<path:path>')
def get_image(path):
    # Safely sends the specific requested image path
    return send_file(BASE_DIR / path)

@app.route('/annotate', methods=['POST'])
def annotate():
    web_path = request.form.get('web_path')
    label = request.form.get('label')
    if label != 'Skip':
        save_annotation(web_path, label)
    else:
        save_annotation(web_path, "Skipped")
    return redirect(url_for('index'))

@app.route('/delete', methods=['POST'])
def delete_annotation():
    web_path = request.form.get('web_path')
    annotations = load_annotations()
    if web_path in annotations:
        del annotations[web_path]
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for k, v in sorted(annotations.items()):
                writer.writerow([k, v])
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting local Flask annotation server...")
    print("Open http://127.0.0.1:5000 in your web browser.")
    app.run(debug=True, port=5000)