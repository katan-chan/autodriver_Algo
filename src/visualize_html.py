"""Export multiple Plotly figures to a single HTML file with tabs."""

import os
from typing import List, Tuple
import plotly.graph_objects as go


def export_figures_to_tabbed_html(
    figures: List[Tuple[str, object]],
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
    
    dropdown_scripts = []
    for i, (tab_name, fig_data) in enumerate(figures):
        content_html = ""
        if isinstance(fig_data, dict) and fig_data.get("type") == "dropdown":
            dropdown_id = f"dropdown-{i}"
            options = fig_data.get("options", [])
            option_buttons = []
            option_divs = []
            for j, (opt_name, opt_fig) in enumerate(options):
                fig_html = opt_fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{i}-{j}")
                active_cls = "active" if j == 0 else ""
                option_buttons.append(
                    f"<button class='dropdown-option {active_cls}' data-target='tab-{i}-opt-{j}'>" f"{opt_name}</button>"
                )
                style = "display:block" if j == 0 else "display:none"
                option_divs.append(
                    f"<div id='tab-{i}-opt-{j}' class='dropdown-panel' style='{style}'>" f"{fig_html}</div>"
                )
            content_html = f"""
            <div class="dropdown-wrapper" id="{dropdown_id}">
                <div class="dropdown-buttons">
                    {''.join(option_buttons)}
                </div>
                {''.join(option_divs)}
            </div>
            """
            dropdown_scripts.append(
                f"initDropdown('{dropdown_id}');"
            )
        else:
            fig = fig_data if isinstance(fig_data, go.Figure) else fig_data["figure"]
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig-{i}")
            content_html = fig_html

        fig_divs.append(f"""
        <div id="tab-{i}" class="tab-content">
            {content_html}
        </div>
        """)

        active_class = "active" if i == 0 else ""
        tab_buttons.append(f"""
            <button class="tab-btn {active_class}" onclick="openTab(event, 'tab-{i}')">{tab_name}</button>
        """)
    
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

        .dropdown-wrapper {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .dropdown-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}

        .dropdown-option {{
            padding: 8px 16px;
            border: 1px solid #ccc;
            background-color: #fafafa;
            border-radius: 4px;
            cursor: pointer;
        }}

        .dropdown-option.active {{
            background-color: #1976d2;
            color: white;
            border-color: #0f5ba5;
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

        function initDropdown(wrapperId) {{
            var wrapper = document.getElementById(wrapperId);
            if (!wrapper) return;
            var buttons = wrapper.getElementsByClassName('dropdown-option');
            for (var i = 0; i < buttons.length; i++) {{
                buttons[i].addEventListener('click', function(evt) {{
                    var targetId = this.getAttribute('data-target');
                    var panels = wrapper.getElementsByClassName('dropdown-panel');
                    for (var j = 0; j < panels.length; j++) {{
                        panels[j].style.display = 'none';
                    }}
                    wrapper.querySelectorAll('.dropdown-option').forEach(function(btn) {{
                        btn.classList.remove('active');
                    }});
                    document.getElementById(targetId).style.display = 'block';
                    this.classList.add('active');
                    setTimeout(function() {{
                        var plotDivs = document.querySelectorAll('#' + targetId + ' .plotly-graph-div');
                        plotDivs.forEach(function(plotDiv) {{
                            Plotly.Plots.resize(plotDiv);
                            Plotly.relayout(plotDiv, {{}});
                        }});
                    }}, 100);
                }});
            }}
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

            // Initialize dropdowns
            {''.join(dropdown_scripts)}
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
