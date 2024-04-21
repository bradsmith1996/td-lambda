## About this Repository
- Author: Peter Bradley Smith
- Implementation timeframe: 2 weeks
- This code was developed to study the 1988 Sutton paper "Learning to Predict by the Methods of Temporal Differences". See the report of results [here](Report.pdf)

## Modules
- main.py:
   - This is the main script for this set of tools. Running 'python main.py' in the terminal will run and generate all 3 graphs that are similar to those generated in Sutton's 1988 paper.
   - main.py includes a class called 'run_training_on_data_set' where
   the different training algorithms are implemented for a given input training data set 'random_walk_sequence', which is also defined in the same file
   - The primary controls of main.py are at the top of the main function, where the user can control which studies to run (figure3, figure4, figure5 which parallel the figure numbers from the Sutton paper)
   - A unit test was also created to test the rms call (rms_unit_test flag)
   - There is a debug flag to give more output details (debug flag)
   - There also and 'inspect_plots' flag, which will give the option to display the output figures
   - After main is ran, new figures will be generated and named according to their reference to the Sutton 1988 paper
   - NOTE: This script, although will generate the same results as the prepared replication paper for Sutton's work, is considered a sandbox, and not necessarily in a finalized state nor considered a released "version"
- create_training_data.py:
   - Generator for training data
   - 'restrict' will allow the user to put distribution weightings within each training data set for the percentage of either left side or right side terminating bounded random walks. This allowed some studies to be performed into the effects of strictly making 50% distributions to be investigated, which was determined to be important for replicating Sutton's results more consistently.
   - This script is also in a working state and not finalized
   - Generated training data is placed in training_data directory
- sources:
   - Contains Suttons original work
- docs:
   - Contains the report word doc on results