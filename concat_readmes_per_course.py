import os

for folder in ['1_NeuralNetworksAndDeepLearning','2_improvingDeepNeuralNetworks','3_StructuringMachineLearningProjects','4_ConvolutionalNeuralNetworks','5_SequenceModels']:
    week_folders = [f for f in os.listdir(folder) if f.lower().startswith('week')]
    big_README = ''
    for wf in week_folders:
        readme_wk_fpath = os.path.join(folder, wf, 'README.md')
        with open(readme_wk_fpath) as f:
            big_README += ''.join(f.readlines())
            big_README += '\n\n'
    with open(os.path.join(folder, 'README.md'), 'w+') as f:
        f.write(big_README)
