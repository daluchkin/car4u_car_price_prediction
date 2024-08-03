#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------------------------------------
todolist.py
------------------------------------------------------------------------------------------------------------
This module contains the ToDoList class, which represents a ToDo list to manage tasks during data analysis 
and track their status. The ToDoList class provides functionality to add tasks, mark them as completed and 
display the list of tasks with their statuses.

Classes:
    ToDoList: A class to manage a list of tasks and their statuses.

Usage Example:
    from todolist import ToDoList
    
    # Create a new ToDo list
    todo_list = ToDoList()
    
    # Add tasks
    todo_list.todo("Data Exploration", "Understand data structure.")
    todo_list.todo("Analyze Data", "Estimate and visualize the distribution of a variable in a dataset.")
    
    # Mark a task as completed
    todo_list.done(1)
    todo_list.done(2)
    
    # Display all tasks
    todo_list.report()
    todo_list.report(status="done)
    todo_list.report(tasklist=[1,3,4])

    # Plot all tasks
    todo_list.plot()

    # Save list
    todo_list.save()
------------------------------------------------------------------------------------------------------------
Author: Dmitry Luchkin
Version: 1.0
Date: 2024-07-26
------------------------------------------------------------------------------------------------------------
"""

import os

import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
import seaborn as sns

TODO_LIST_FILE_NAME = '../00_data/02_todo/todo_list.csv'

class ToDoList():
    """
    A class to represent a ToDo list for managing tasks and tracking their status.

    Methods:
        open()
            Opens saved list of tasks.
        save()
            Saves ToDo list into CSV file TODO_LIST_FILE_NAME
        todo(phase:str, task:str)
            Adds a new task to the list.
        done(index:int)
            Sets 'done' status for the task.
        report(self, status:str='all|undone|done')
            Creates HTML table of ToDo list with status.
        plot()
            Create a plot of the pivot table showing task statuses.
    """

    
    def __init__(self, new: bool=True):
        """
        Initializes the ToDoList object.

        Args:
            new (bool): If True, then a new list will be created, otherwise the list will be loaded from TODO_LIST_FILE_NAME
        """
        if new:
            self._todo_list = pd.DataFrame(columns=['phase', 'task', 'status'])
        elif os.path.exists(TODO_LIST_FILE_NAME):
                self.open()
        else:
            raise FileNotFoundError(f'File {TODO_LIST_FILE_NAME} has not found.')

    
    def open(self):
        """
        Opens saved list of tasks. 
        
        Retruns:
            str: HTML code of table with status of the tasks.    
        """
        self._todo_list = pd.read_csv(TODO_LIST_FILE_NAME)
        return self.report(status='all')

    
    def save(self):
        """
        Saves ToDo list into CSV file ../00_data/02_todo/todo_list.csv
        """
        
        self._todo_list.to_csv('../00_data/02_todo/todo_list.csv')

            
    def todo(self, phase:str, task:str) -> None:
        """
        Adds a new task to the list.

        Args:
            phase (str): The name of the phase in which the task was created.
            task (str): The description of the task.
        """
        
        self._todo_list.loc[self._todo_list.shape[0]] = {'phase': phase, 'task': task, 'status': 'undone'}

    
    def done(self, index:int):
        """
        Sets 'done' status for the task.

        Args:
            index (int): A sequentual number of the task.
        """
        
        if 0 <= index-1 < len(self._todo_list):
            self._todo_list.at[index-1, 'status'] = 'done'
        else:
            print(f"Task at index {index} does not exist.")

    
    def report(self, status:str='all', tasklist=None):
        """
        Creates HTML table of ToDo list with status.

        Args:
            status (str): A mode of output of the list.
                'all': Entire ToDo list.
                'undone': The list of undone tasks only.
                'done': The list of done tasks only.
            tasklist ([int]): A list of numbers of tasks. It is None by default.

        Returns:
            str: HTML code of table with status of the tasks. 
        """
        
        html = '''
            <style type="text/css">
            .tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;width: 100%;}
            .tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
              font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
            .tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
              font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
            .tg .tg-92ng{background-color:#efefef;border-color:inherit;font-weight:bold;position:-webkit-sticky;position:sticky;
              text-align:left !important; top:-1px;vertical-align:top;will-change:transform;max-width:100%;white-space:nowrap;}
            .tg .tg-0pky{border-color:inherit;text-align:left !important;vertical-align:top}
            .tg .tg-done{border-color:inherit;text-align:center !important;vertical-align:top}
            .col1 { width: 50px; }
            .col2 { width: 150px; }
            .col3 { width: auto; }
            .col4 { width: 100px;}
            </style>
            <p><strong>ToDo List</strong></p>
            <table class="tg"><thead>
              <col class="col1"/>
              <col class="col2"/>
              <col class="col3"/>
              <col class="col4"/>
              <tr>
                <th class="tg-92ng">#</th>
                <th class="tg-92ng">Phase</th>
                <th class="tg-92ng">Task</th>
                <th class="tg-92ng">Status</th>
              </tr></thead>
            <tbody>
                {list}
            </tbody>
            </table>
            '''
        todos = ''
        for index, row in self._todo_list.iterrows():    
            if status == 'all' or row['status'] == status:
                # check indexes
                if tasklist is not None and index+1 not in tasklist:
                    continue
                
                todos += f'''
                    <tr>
                    <td class="tg-0pky">{index+1}</td>
                    <td class="tg-0pky">{row['phase']}</td>
                    <td class="tg-0pky">{row['task']}</td>
                    <td class="tg-done">{'🟢' if row['status'] == 'done' else '🔴'}</td>
                    </tr>
                    '''
        html = html.replace('{list}', todos)
        return HTML(html)

    
    def plot(self):
        """
        Creates a plot of the pivot table showing task statuses.
        """
        
        plt.figure(figsize = (15, 4))
        df_plot = todo_list._todo_list.groupby(['phase', 'status']).size().reset_index().pivot(columns='status', index='phase', values=0)
        df_plot.plot(kind='bar', stacked=True, color=['green', 'red'])
        plt.ylabel('Count')
        plt.xlabel('Phase')
        plt.legend(title='Status')
        plt.title(f'ToDo List Status')
        plt.show()
