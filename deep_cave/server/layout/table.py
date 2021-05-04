import dash_table


table = dash_table.DataTable(
        id='table_of_studies',
        data=[],
        style_table={'overflowY': 'auto', 'overflowX': 'auto'},
        fixed_rows={'headers': True},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_cell={
            'minWidth': 100, 'maxWidth': 400
        },
        merge_duplicate_headers=True,
        page_action="native",
        page_current=0,
        page_size=10,
        filter_action='native',
        sort_action='native',
        selected_columns=[],
        editable=True,
        column_selectable='multi',
        css=[{'selector': '.row', 'rule': 'margin: 0'}]
    )
