Ball positions were labeled manually using a set of self-written tools.

The file format of labels is a CSV files with an index, corresponding to the frame_number in the video, with an x and y column representing the ball position in the video.

labeler.py is the tool for ab initio labeling of video files. Labels can be viewed using the show_labels.py CLI tool. 
Existing labels can be edited using the postprocessing.py CLI tool. It allows to delete and insert ball positions in the data set on a per-frame resolution while viewing the annotated data set.

A caveat worth mentioning is that jumping back and forth in videostreams using openCV does not work as expected, so we had to implement a circle buffer to allow jumping back while labeling. Take-home message: Be careful with videostreams in openCV.

For further information, please read the source code (the function of control keys can be easily derived there). If you encounter any issues and/or need further information, please contact moritz.schaefer@biol.ethz.ch and m.morik@tu-berlin.de.
