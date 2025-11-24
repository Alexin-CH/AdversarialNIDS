def plot_classification_report(report_dict, ax, title):
    """
    Plot classification report as a formatted table.
    
    Args:
        report_dict: Classification report dictionary from sklearn
        ax: Matplotlib axis to plot on
        title: Table title
    """
    metrics = ['precision', 'recall', 'f1-score', 'support']
    
    # Get class names (exclude aggregate metrics)
    class_names = [k for k in report_dict.keys() 
                   if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Build table data
    data = []
    for class_name in class_names:
        row = [class_name] + [
            f"{report_dict[class_name][m]:.4f}" if m != 'support' 
            else str(int(report_dict[class_name][m])) 
            for m in metrics
        ]
        data.append(row)
    
    # Add separator
    data.append(['---'] * 5)
    
    # Add macro average
    if 'macro avg' in report_dict:
        row = ['macro avg'] + [
            f"{report_dict['macro avg'][m]:.4f}" if m != 'support' 
            else str(int(report_dict['macro avg'][m])) 
            for m in metrics
        ]
        data.append(row)
    
    # Add weighted average
    if 'weighted avg' in report_dict:
        row = ['weighted avg'] + [
            f"{report_dict['weighted avg'][m]:.4f}" if m != 'support' 
            else str(int(report_dict['weighted avg'][m])) 
            for m in metrics
        ]
        data.append(row)
    
    # Add accuracy
    if 'accuracy' in report_dict:
        acc_row = ['accuracy', '', '', f"{report_dict['accuracy']:.4f}", 
                   str(int(report_dict['weighted avg']['support']))]
        data.append(acc_row)
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=data,
        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.2, 0.2, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style separator row
    separator_idx = len(class_names) + 1
    for i in range(5):
        table[(separator_idx, i)].set_facecolor('#E7E6E6')
    
    ax.set_title(title, fontsize=12, weight='bold', pad=50)