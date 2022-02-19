package com.gui;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;

public class Trainer extends JFrame {
    JMenu file_menu;
    JMenu edit_menu;
    JMenu help_menu;

    JMenu model_menu;
    JMenu dataset_menu;
    JMenu classification_ds_menu;
    JMenu regression_ds_menu;

    JMenuItem load_model_menu;
    JMenuItem save_model_menu;
    JMenuItem create_model_menu;

    JMenuItem load_classification_dataset_menu;
    JMenuItem load_regression_dataset_menu;
    JMenu default_classification_ds_menu;
    JMenu default_regression_ds_menu;
    JMenuItem digits_dataset_menu;
    JMenuItem clusters_dataset_menu;
    JMenuItem simple_regression_dataset_menu;

    JMenuItem about_menu;

    JMenuBar menuBar;

    public Trainer() {
        menuBar = new JMenuBar();
        file_menu = new JMenu("File");
        edit_menu = new JMenu("Edit");
        model_menu = new JMenu("Model");
        dataset_menu = new JMenu("Dataset");
        classification_ds_menu = new JMenu("Classification");
        regression_ds_menu = new JMenu("Regression");

        load_model_menu = new JMenuItem("Load Model");
        create_model_menu = new JMenuItem("Create Model");
        save_model_menu = new JMenuItem("Save Model");

        load_classification_dataset_menu = new JMenuItem("Load");
        load_regression_dataset_menu = new JMenuItem("Load");
        default_classification_ds_menu = new JMenu("Default");
        default_regression_ds_menu = new JMenu("Default");
        digits_dataset_menu = new JMenuItem("DIGITS");
        clusters_dataset_menu = new JMenuItem("random CLUSTERS");
        simple_regression_dataset_menu = new JMenuItem("random REGRESSION");

        model_menu.add(create_model_menu);
        model_menu.add(load_model_menu);
        model_menu.add(save_model_menu);

        file_menu.add(model_menu);

        default_classification_ds_menu.add(digits_dataset_menu);
        default_classification_ds_menu.add(clusters_dataset_menu);
        default_regression_ds_menu.add(simple_regression_dataset_menu);
        classification_ds_menu.add(load_classification_dataset_menu);
        classification_ds_menu.add(default_classification_ds_menu);
        regression_ds_menu.add(load_regression_dataset_menu);
        regression_ds_menu.add(default_regression_ds_menu);
        dataset_menu.add(classification_ds_menu);
        dataset_menu.add(regression_ds_menu);

        file_menu.add(dataset_menu);

        about_menu = new JMenuItem("About");
        help_menu = new JMenu("Help");
        help_menu.add(about_menu);

        menuBar.add(file_menu);
        menuBar.add(edit_menu);
        menuBar.add(help_menu);

        this.add(menuBar, "North");
        setTitle("Trainer");
        setSize(800, 600);
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public static void main(String args[]) {
        new Trainer();
    }
}