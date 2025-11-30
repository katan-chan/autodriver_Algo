"""Export multiple Plotly figures to a single HTML file with tabs."""

import os
from typing import List, Tuple
import plotly.graph_objects as go


def export_figures_to_tabbed_html(
    figures: List[Tuple[str, go.Figure]],
    output_path: str,
    title: str = "Traffic Simulation Results",
) -> None:
    """
    Export multiple Plotly figures to a single HTML file with tabs.
    
    Args:
        figures: List of (tab_name, figure) tuples
        output_path: Path to save HTML file
        title: Page title
    """
    
    # Generate HTML for each figure
    fig_divs = []
    tab_buttons = []
    
    for i, (tab_name, fig) in enumerate(figures):
        # Get the inner HTML of the figure (without full HTML wrapper)
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=f"fig-{i}",
        )
        
        # Create tab content div - all visible initially, will hide via JS
        fig_divs.append(f'''
        <div id="tab-{i}" class="tab-content">
            {fig_html}
        </div>
        ''')
        
        # Create tab button
        active_class = "active" if i == 0 else ""
        tab_buttons.append(f'''
            <button class="tab-btn {active_class}" onclick="openTab(event, 'tab-{i}')">{tab_name}</button>
        ''')
    
    # Full HTML template
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
        
        .tab-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 20px;
            background-color: #fff;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .tab-btn {{
            padding: 10px 20px;
            border: none;
            background-color: #e0e0e0;
            cursor: pointer;
            border-radius: 4px;
            font-size: 14px;
            transition: all 0.2s;
        }}
        
        .tab-btn:hover {{
            background-color: #d0d0d0;
        }}
        
        .tab-btn.active {{
            background-color: #4CAF50;
            color: white;
        }}
        
        .tab-content {{
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .summary {{
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="tab-container">
        {''.join(tab_buttons)}
    </div>
    
    {''.join(fig_divs)}
    
    <script>
        function openTab(evt, tabId) {{
            // Hide all tab contents
            var tabContents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < tabContents.length; i++) {{
                tabContents[i].style.display = "none";
            }}
            
            // Remove active class from all buttons
            var tabBtns = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < tabBtns.length; i++) {{
                tabBtns[i].classList.remove("active");
            }}
            
            // Show current tab and add active class
            document.getElementById(tabId).style.display = "block";
            evt.currentTarget.classList.add("active");
            
            // Trigger Plotly resize for proper rendering (with delay)
            setTimeout(function() {{
                var plotDivs = document.querySelectorAll('#' + tabId + ' .plotly-graph-div');
                plotDivs.forEach(function(plotDiv) {{
                    Plotly.Plots.resize(plotDiv);
                    Plotly.relayout(plotDiv, {{}});
                }});
            }}, 100);
        }}
        
        // After page loads: resize all plots, then hide non-active tabs
        window.onload = function() {{
            // First, resize all plots while visible
            var plotDivs = document.querySelectorAll('.plotly-graph-div');
            plotDivs.forEach(function(plotDiv) {{
                Plotly.Plots.resize(plotDiv);
            }});
            
            // Then hide all tabs except first
            setTimeout(function() {{
                var tabContents = document.getElementsByClassName("tab-content");
                for (var i = 1; i < tabContents.length; i++) {{
                    tabContents[i].style.display = "none";
                }}
            }}, 500);
        }};
    </script>
</body>
</html>
'''
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Đã lưu visualization vào: {output_path}")
    
    # Open in browser
    import webbrowser
    webbrowser.open('file://' + os.path.abspath(output_path))
