from .dev import util
from .model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
# import matplotlib.pyplot as plt

# src_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 6.0: 6, 7.0: 7, 8.0: 8, 9.0: 9, 10.0: 10, 11.0: 11, 12.0: 12, 13.0: 13, 14.0: 14, 15.0: 15, 16.0: 16, 17.0: 17, 18.0: 18, 19.0: 19, 20.0: 20, 21.0: 21, 22.0: 22, 23.0: 23, 24.0: 24, 25.0: 25, 26.0: 26, 27.0: 27, 28.0: 28, 29.0: 29, 30.0: 30, 31.0: 31, 32.0: 32, 33.0: 33, 34.0: 34, 35.0: 35, 36.0: 36, 37.0: 37, 38.0: 38, 39.0: 39, 40.0: 40, 41.0: 41, 42.0: 42, 43.0: 43, 44.0: 44, 45.0: 45, 46.0: 46, 47.0: 47, 48.0: 48, 49.0: 49,
#              50.0: 50, 51.0: 51, 52.0: 52, 53.0: 53, 54.0: 54, 55.0: 55, 56.0: 56, 57.0: 57, 58.0: 58, 59.0: 59, 60.0: 60, 61.0: 61, 62.0: 62, 63.0: 63, 64.0: 64, 65.0: 65, 66.0: 66, 67.0: 67, 68.0: 68, 69.0: 69, 70.0: 70, 71.0: 71, 72.0: 72, 73.0: 73, 74.0: 74, 75.0: 75, 76.0: 76, 77.0: 77, 78.0: 78, 79.0: 79, 80.0: 80, 81.0: 81, 82.0: 82, 83.0: 83, 84.0: 84, 85.0: 85, 86.0: 86, 87.0: 87, 88.0: 88, 89.0: 89, 90.0: 90, 91.0: 91, 92.0: 92, 93.0: 93, 94.0: 94, 95.0: 95, 96.0: 96, 97.0: 97, 98.0: 98, 99.0: 99}

# src_interactions = [
#     [55, 57], [78, 68], [98, 46], [19, 0], [55, 84], [28, 76], [83, 21], [41, 15], [21, 46], [0, 23], [48, 72], [21, 64], [84, 13], [52, 42], [56, 3], [85, 48], [10, 54], [39, 99], [97, 31], [37, 35], [38, 9], [98, 23], [78, 54], [70, 50], [99, 86], [67, 63], [24, 92], [28, 53], [90, 37], [47, 66], [90, 46], [63, 38], [41, 10], [41, 19], [21, 50], [44, 15], [29, 63], [44, 24], [73, 69], [34, 16], [10, 58], [3, 19], [24, 60], [70, 27], [36, 52], [99, 90], [59, 63], [17, 57], [48, 35], [5, 64], [83, 11], [80, 76], [6, 47], [40, 58], [33, 19], [73, 55], [10, 35], [62, 73], [85, 38], [93, 51], [0, 98], [2, 40], [85, 56], [95, 88], [12, 81], [86, 39], [87,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 93], [35, 64], [13, 73], [7, 2], [89, 53], [57, 79], [16, 87], [71, 32], [21, 13], [28, 61], [63, 28], [58, 71], [5, 77], [18, 96], [41, 9], [39, 48], [58, 98], [64, 1], [85, 33], [20, 71], [95, 74], [85, 51], [13, 32], [3, 18], [59, 17], [78, 30], [24, 59], [70, 35], [98, 17], [28, 38], [48, 16], [46, 83], [69, 48], [76, 96], [46, 92], [5, 63], [58, 66], [26, 95], [49, 97], [42, 58], [30, 65], [10, 7], [2, 3], [39, 52], [32, 22], [43, 22], [51, 44], [3, 4], [4, 58], [35, 45], [59, 12], [97, 29], [17, 6], [47, 19], [96, 79], [36, 37], [5, 31], [57, 69], [14, 98], [77, 47], [5, 40], [29, 16], [81, 17], [26, 81], [49, 92], [61, 57], [73, 22], [54, 27], [93, 18], [23, 39], [8, 87], [43, 17], [24, 22], [1, 29], [24, 40], [25, 94], [35, 40], [43, 53], [67, 20], [1, 47], [70, 16], [96, 65], [1, 56], [88, 61], [1, 65], [92, 22], [17, 28], [37, 95], [28, 37], [62, 17], [18, 81], [8, 46], [69, 65], [7, 90], [20, 38], [20, 56], [60, 92], [23, 52], [66, 32], [70, 2], [1, 42], [25, 98], [5, 3], [3, 70], [80, 24], [80, 33], [69, 42], [42, 25], [26, 71], [81, 7], [90, 83], [11, 37], [20, 15], [8, 59], [54, 26], [74, 4], [95, 36], [72, 43], [51, 20], [60, 87], [52, 83], [72, 61], [95, 63], [53, 57], [95, 72], [53, 66], [72, 79], [1, 37], [17, 0], [68, 63], [7, 44], [77, 41], [57, 72], [42, 20], [50, 33], [7, 62], [26, 75], [79, 78], [30, 36], [18, 80], [94, 48], [23, 6], [8, 54], [60, 64], [0, 59], [39, 41], [4, 29], [35, 16], [99, 19], [10, 81], [45, 48], [35, 25], [99, 28], [57, 31], [25, 97], [49, 36], [22, 91], [57, 58], [18, 39], [68, 58], [97, 94], [61, 19], [98, 59], [70, 86], [80, 32], [55, 97], [61, 28], [78, 99], [18, 57], [18, 66], [71, 69], [0, 27], [11, 27], [19, 49], [82, 96], [54, 25], [84, 26], [51, 1], [29, 99], [4, 24], [25, 56], [64, 38], [72, 51], [99, 23], [51, 95], [85, 97], [49, 22], [77, 4], [38, 31], [13, 96], [78, 76], [69, 18], [69, 27], [30, 8], [98, 63], [90, 59], [0, 4], [8, 17], [30, 17], [28, 84], [90, 68], [7, 61], [48, 62], [91, 33], [11, 22], [48, 71], [29, 67], [71, 82], [52, 41], [12, 14], [75, 52], [83, 65], [12, 23], [41, 59], [43, 1], [21, 99], [99, 18], [57, 21], [32, 86], [56, 65], [77, 8], [49, 35], [70, 67], [49, 44], [42,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           5], [71, 50], [18, 56], [92, 91], [39, 8], [60, 58], [4, 5], [72, 23], [84, 25], [62, 95], [52, 72], [93, 73], [37, 29], [65, 11], [3, 36], [2, 80], [43, 81], [66, 83], [24, 77], [32, 90], [67, 57], [24, 86], [15, 28],
#     [49, 30], [71, 27], [59, 71], [47, 78], [42, 9], [59, 80], [47, 87], [48, 52], [36, 96], [0, 21], [75, 24], [84, 11], [23, 4], [93, 50], [25, 41], [53, 23], [74, 64], [97, 38], [93, 86], [57, 11], [86, 47], [74, 91], [35, 72], [97, 74], [18, 28], [36, 73], [47, 82], [77, 83], [5, 76], [92, 90], [52, 17], [41, 44], [21, 75], [73, 76], [53, 18], [93, 54], [20, 79], [78, 29], [38, 2], [51, 84], [24, 67], [59, 34], [43, 98], [48, 15], [17, 46], [16, 90], [17, 55], [59, 70], [5, 62], [71, 44], [77, 78], [6, 36], [92, 76], [48, 51], [83, 18], [41, 12], [29, 56], [60, 34], [10, 15], [52, 30], [42, 93], [81, 84], [2, 29], [73, 80], [31, 83], [32, 48], [54,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  85], [85, 63], [54, 94], [85, 72], [68, 1], [97, 46], [24, 71], [82, 3], [1, 78], [55, 67], [47, 63], [5, 66], [83, 13], [29, 51], [75, 18], [69, 96], [73, 66], [2, 33], [32, 43], [4, 70], [43, 52], [12, 83], [23, 83],
#     [66, 63], [96, 73], [47, 31], [98, 15], [17, 27], [90, 11], [7, 4], [16, 71], [58, 64], [6, 35], [92, 84], [27, 76], [73, 43], [20, 46], [25, 12], [62, 70], [3, 2], [4, 56], [24, 34], [31, 82], [4, 65], [32, 65], [34, 84], [
#         55, 30], [96, 77], [71, 2], [92, 43], [77, 54], [29, 14], [92, 52], [68, 94], [7, 84], [8, 58], [42, 69], [54, 52], [67, 0], [51, 46], [97, 13], [14, 6], [12, 73], [16, 34], [53, 92], [88, 68], [69, 45], [61, 41],
#     [42, 37], [58, 63], [62, 33], [7, 97], [18, 97], [10, 4], [73, 42], [54, 38], [66, 3], [54, 47], [0, 76], [4, 37], [72, 64], [32, 37], [55, 20], [5, 1], [17, 3], [96, 58], [68, 48], [56, 92], [69, 22], [14, 86], [58, 40], [92, 42], [39, 17], [71, 86], [79, 99], [0, 44], [42, 59], [63, 91], [83, 69], [93, 15], [74, 11], [39, 53], [20, 49], [75, 83], [97, 3], [33, 86], [53, 64], [44, 95], [55, 6], [25, 91], [65, 47], [57, 43], [69, 17], [3, 90], [92, 37], [70, 98], [98, 80], [26, 73], [7, 69], [8, 34], [54, 1], [71, 90], [39, 30], [23, 13], [93, 10], [29, 93], [51, 4], [60, 71], [23, 31], [83, 91], [84, 56], [35, 14], [4, 45], [1, 30], [37, 51], [53,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  77], [37, 60], [3, 67], [3, 76], [70, 84], [14, 94], [8, 29], [39, 16], [6, 86], [39, 34], [75, 64], [43, 4], [99, 12], [41, 89], [25, 72], [37, 46], [99, 30], [68, 42], [82, 53], [8, 24], [42, 35], [12, 3], [48, 87], [63, 85], [95, 28], [29, 92], [33, 53], [12, 30], [76, 5], [76, 14], [72, 62], [10, 87], [68, 19], [97, 64], [77, 15], [68, 46], [79, 43], [36, 72], [47, 81], [67, 96], [27, 23], [0, 6], [78, 96], [71, 57], [79, 61], [73, 8], [92, 98], [60, 56], [75, 63], [81, 97], [4, 21], [93, 71], [25, 62], [10, 73], [68, 5], [34, 49], [2, 78], [27, 9], [78, 91], [27, 18], [59, 87], [48, 50], [17, 81], [77, 95], [0, 28], [83, 53], [2, 46], [62, 97], [64, 39], [85, 80], [51, 87], [13, 70], [26, 12], [97, 63]
# ]
# tar_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5, 6.0: 6, 7.0: 7, 8.0: 8, 9.0: 9, 10.0: 10, 11.0: 11, 12.0: 12, 13.0: 13, 14.0: 14, 15.0: 15, 16.0: 16, 17.0: 17, 18.0: 18, 19.0: 19, 20.0: 20, 21.0: 21, 22.0: 22, 23.0: 23, 24.0: 24, 25.0: 25, 26.0: 26, 27.0: 27, 28.0: 28, 29.0: 29, 30.0: 30, 31.0: 31, 32.0: 32, 33.0: 33, 34.0: 34, 35.0: 35, 36.0: 36, 37.0: 37, 38.0: 38, 39.0: 39, 40.0: 40, 41.0: 41, 42.0: 42, 43.0: 43, 44.0: 44, 45.0: 45, 46.0: 46, 47.0: 47, 48.0: 48, 49.0: 49,
#              50.0: 50, 51.0: 51, 52.0: 52, 53.0: 53, 54.0: 54, 55.0: 55, 56.0: 56, 57.0: 57, 58.0: 58, 59.0: 59, 60.0: 60, 61.0: 61, 62.0: 62, 63.0: 63, 64.0: 64, 65.0: 65, 66.0: 66, 67.0: 67, 68.0: 68, 69.0: 69, 70.0: 70, 71.0: 71, 72.0: 72, 73.0: 73, 74.0: 74, 75.0: 75, 76.0: 76, 77.0: 77, 78.0: 78, 79.0: 79, 80.0: 80, 81.0: 81, 82.0: 82, 83.0: 83, 84.0: 84, 85.0: 85, 86.0: 86, 87.0: 87, 88.0: 88, 89.0: 89, 90.0: 90, 91.0: 91, 92.0: 92, 93.0: 93, 94.0: 94, 95.0: 95, 96.0: 96, 97.0: 97, 98.0: 98, 99.0: 99}
# tar_interactions = [
#     [55, 57], [78, 68], [98, 46], [19, 0], [55, 84], [28, 76], [83, 21], [41, 15], [21, 46], [0, 23], [48, 72], [21, 64], [84, 13], [52, 42], [56, 3], [85, 48], [10, 54], [39, 99], [97, 31], [37, 35], [38, 9], [98, 23], [78, 54], [70, 50], [99, 86], [67, 63], [24, 92], [28, 53], [90, 37], [47, 66], [90, 46], [63, 38], [41, 10], [41, 19], [21, 50], [44, 15], [29, 63], [44, 24], [73, 69], [34, 16], [10, 58], [3, 19], [24, 60], [70, 27], [36, 52], [99, 90], [59, 63], [17, 57], [48, 35], [5, 64], [83, 11], [80, 76], [6, 47], [40, 58], [33, 19], [73, 55], [10, 35], [62, 73], [85, 38], [93, 51], [0, 98], [2, 40], [85, 56], [95, 88], [12, 81], [86, 39], [87,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 93], [35, 64], [13, 73], [7, 2], [89, 53], [57, 79], [16, 87], [71, 32], [21, 13], [28, 61], [63, 28], [58, 71], [5, 77], [18, 96], [41, 9], [39, 48], [58, 98], [64, 1], [85, 33], [20, 71], [95, 74], [85, 51], [13, 32], [3, 18], [59, 17], [78, 30], [24, 59], [70, 35], [98, 17], [28, 38], [48, 16], [46, 83], [69, 48], [76, 96], [46, 92], [5, 63], [58, 66], [26, 95], [49, 97], [42, 58], [30, 65], [10, 7], [2, 3], [39, 52], [32, 22], [43, 22], [51, 44], [3, 4], [4, 58], [35, 45], [59, 12], [97, 29], [17, 6], [47, 19], [96, 79], [36, 37], [5, 31], [57, 69], [14, 98], [77, 47], [5, 40], [29, 16], [81, 17], [26, 81], [49, 92], [61, 57], [73, 22], [54, 27], [93, 18], [23, 39], [8, 87], [43, 17], [24, 22], [1, 29], [24, 40], [25, 94], [35, 40], [43, 53], [67, 20], [1, 47], [70, 16], [96, 65], [1, 56], [88, 61], [1, 65], [92, 22], [17, 28], [37, 95], [28, 37], [62, 17], [18, 81], [8, 46], [69, 65], [7, 90], [20, 38], [20, 56], [60, 92], [23, 52], [66, 32], [70, 2], [1, 42], [25, 98], [5, 3], [3, 70], [80, 24], [80, 33], [69, 42], [42, 25], [26, 71], [81, 7], [90, 83], [11, 37], [20, 15], [8, 59], [54, 26], [74, 4], [95, 36], [72, 43], [51, 20], [60, 87], [52, 83], [72, 61], [95, 63], [53, 57], [95, 72], [53, 66], [72, 79], [1, 37], [17, 0], [68, 63], [7, 44], [77, 41], [57, 72], [42, 20], [50, 33], [7, 62], [26, 75], [79, 78], [30, 36], [18, 80], [94, 48], [23, 6], [8, 54], [60, 64], [0, 59], [39, 41], [4, 29], [35, 16], [99, 19], [10, 81], [45, 48], [35, 25], [99, 28], [57, 31], [25, 97], [49, 36], [22, 91], [57, 58], [18, 39], [68, 58], [97, 94], [61, 19], [98, 59], [70, 86], [80, 32], [55, 97], [61, 28], [78, 99], [18, 57], [18, 66], [71, 69], [0, 27], [11, 27], [19, 49], [82, 96], [54, 25], [84, 26], [51, 1], [29, 99], [4, 24], [25, 56], [64, 38], [72, 51], [99, 23], [51, 95], [85, 97], [49, 22], [77, 4], [38, 31], [13, 96], [78, 76], [69, 18], [69, 27], [30, 8], [98, 63], [90, 59], [0, 4], [8, 17], [30, 17], [28, 84], [90, 68], [7, 61], [48, 62], [91, 33], [11, 22], [48, 71], [29, 67], [71, 82], [52, 41], [12, 14], [75, 52], [83, 65], [12, 23], [41, 59], [43, 1], [21, 99], [99, 18], [57, 21], [32, 86], [56, 65], [77, 8], [49, 35], [70, 67], [49, 44], [42,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           5], [71, 50], [18, 56], [92, 91], [39, 8], [60, 58], [4, 5], [72, 23], [84, 25], [62, 95], [52, 72], [93, 73], [37, 29], [65, 11], [3, 36], [2, 80], [43, 81], [66, 83], [24, 77], [32, 90], [67, 57], [24, 86], [15, 28],
#     [49, 30], [71, 27], [59, 71], [47, 78], [42, 9], [59, 80], [47, 87], [48, 52], [36, 96], [0, 21], [75, 24], [84, 11], [23, 4], [93, 50], [25, 41], [53, 23], [74, 64], [97, 38], [93, 86], [57, 11], [86, 47], [74, 91], [35, 72], [97, 74], [18, 28], [36, 73], [47, 82], [77, 83], [5, 76], [92, 90], [52, 17], [41, 44], [21, 75], [73, 76], [53, 18], [93, 54], [20, 79], [78, 29], [38, 2], [51, 84], [24, 67], [59, 34], [43, 98], [48, 15], [17, 46], [16, 90], [17, 55], [59, 70], [5, 62], [71, 44], [77, 78], [6, 36], [92, 76], [48, 51], [83, 18], [41, 12], [29, 56], [60, 34], [10, 15], [52, 30], [42, 93], [81, 84], [2, 29], [73, 80], [31, 83], [32, 48], [54,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  85], [85, 63], [54, 94], [85, 72], [68, 1], [97, 46], [24, 71], [82, 3], [1, 78], [55, 67], [47, 63], [5, 66], [83, 13], [29, 51], [75, 18], [69, 96], [73, 66], [2, 33], [32, 43], [4, 70], [43, 52], [12, 83], [23, 83],
#     [66, 63], [96, 73], [47, 31], [98, 15], [17, 27], [90, 11], [7, 4], [16, 71], [58, 64], [6, 35], [92, 84], [27, 76], [73, 43], [20, 46], [25, 12], [62, 70], [3, 2], [4, 56], [24, 34], [31, 82], [4, 65], [32, 65], [34, 84], [
#         55, 30], [96, 77], [71, 2], [92, 43], [77, 54], [29, 14], [92, 52], [68, 94], [7, 84], [8, 58], [42, 69], [54, 52], [67, 0], [51, 46], [97, 13], [14, 6], [12, 73], [16, 34], [53, 92], [88, 68], [69, 45], [61, 41],
#     [42, 37], [58, 63], [62, 33], [7, 97], [18, 97], [10, 4], [73, 42], [54, 38], [66, 3], [54, 47], [0, 76], [4, 37], [72, 64], [32, 37], [55, 20], [5, 1], [17, 3], [96, 58], [68, 48], [56, 92], [69, 22], [14, 86], [58, 40], [92, 42], [39, 17], [71, 86], [79, 99], [0, 44], [42, 59], [63, 91], [83, 69], [93, 15], [74, 11], [39, 53], [20, 49], [75, 83], [97, 3], [33, 86], [53, 64], [44, 95], [55, 6], [25, 91], [65, 47], [57, 43], [69, 17], [3, 90], [92, 37], [70, 98], [98, 80], [26, 73], [7, 69], [8, 34], [54, 1], [71, 90], [39, 30], [23, 13], [93, 10], [29, 93], [51, 4], [60, 71], [23, 31], [83, 91], [84, 56], [35, 14], [4, 45], [1, 30], [37, 51], [53,
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  77], [37, 60], [3, 67], [3, 76], [70, 84], [14, 94], [8, 29], [39, 16], [6, 86], [39, 34], [75, 64], [43, 4], [99, 12], [41, 89], [25, 72], [37, 46], [99, 30], [68, 42], [82, 53], [8, 24], [42, 35], [12, 3], [48, 87], [63, 85], [95, 28], [29, 92], [33, 53], [12, 30], [76, 5], [76, 14], [72, 62], [10, 87], [68, 19], [97, 64], [77, 15], [68, 46], [79, 43], [36, 72], [47, 81], [67, 96], [27, 23], [0, 6], [78, 96], [71, 57], [79, 61], [73, 8], [92, 98], [60, 56], [75, 63], [81, 97], [4, 21], [93, 71], [25, 62], [10, 73], [68, 5], [34, 49], [2, 78], [27, 9], [78, 91], [27, 18], [59, 87], [48, 50], [17, 81], [77, 95], [0, 28], [83, 53], [2, 46], [62, 97], [64, 39], [85, 80], [51, 87], [13, 70], [26, 12], [97, 63]
# ]

# data = {
#     'src_index': src_index,
#     'src_interactions': src_interactions,
#     'tar_index': tar_index,
#     'tar_interactions': tar_interactions,
#     'mutual_interactions': None
# }


def main(data):
    data_name = 'syn'
    result_folder = 'results'
    c = 'cosine'
    m = 'proximal'

    data_mc3 = data

    # connects = np.zeros(
    #     (len(data_mc3['src_index']), len(data_mc3['src_index'])))
    # for item in data_mc3['src_interactions']:
    #     connects[item[0], item[1]] += 1
    # plt.imshow(connects)
    # plt.savefig('{}/{}_src.png'.format(result_folder, data_name))
    # plt.close('all')

    # connects = np.zeros(
    #     (len(data_mc3['tar_index']), len(data_mc3['tar_index'])))
    # for item in data_mc3['tar_interactions']:
    #     connects[item[0], item[1]] += 1
    # plt.imshow(connects)
    # plt.savefig('{}/{}_tar.png'.format(result_folder, data_name))
    # plt.close('all')

    opt_dict = {'epochs': 5,
                'batch_size': 10000,
                'use_cuda': False,
                'strategy': 'soft',
                'beta': 1e-1,
                'outer_iteration': 400,
                'inner_iteration': 1,
                'sgd_iteration': 300,
                'prior': False,
                'prefix': result_folder,
                'display': True}

    hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                      'tar_number': len(data_mc3['tar_index']),
                      'dimension': 20,
                      'loss_type': 'L2',
                      'cost_type': c,
                      'ot_method': m}

    gwd_model = GromovWassersteinLearning(hyperpara_dict)

    # initialize optimizer
    optimizer = optim.Adam(
        gwd_model.gwl_model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer, gamma=0.8)

    print(gwd_model.obtain_embedding(
        opt_dict, torch.LongTensor([0, 1]), 0))

    # Gromov-Wasserstein learning
    gwd_model.train_without_prior(
        data_mc3, optimizer, opt_dict, scheduler=None)
    print(data_mc3)
    print(optimizer)
    print(opt_dict)
    # save model
    gwd_model.save_model(
        '{}/model_{}_{}_{}.pt'.format(result_folder, data_name, m, c))
    gwd_model.save_matching(
        '{}/result_{}_{}_{}.pkl'.format(result_folder, data_name, m, c))

    # emb = gwd_model.gwl_model.emb_model[0]

    # indxx = torch.LongTensor([0, 1])
    # # print(emb(indxx))
    # print(emb)
    print(gwd_model.obtain_embedding(
        opt_dict, torch.LongTensor([0, 1]), 0))
    # print(data_mc3['src_index'].keys())
    # print(data_mc3['src_index'].keys()[0])

    # print(gwd_model.gwl_model.emb_model)
    # print(gwd_model.gwl_model.emb_model[0])
    # # print(gwd_model.gwl_model.emb_model[1])
    # print(gwd_model.d_gw)
    # print(gwd_model.d_gw[0])
    # print(gwd_model.gwl_model.emb_model[0](gwd_model.d_gw[0]))
    # print(gwd_model.gwl_model.emb_model[1](0))