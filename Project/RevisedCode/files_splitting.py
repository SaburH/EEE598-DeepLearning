import os, random, shutil

direc_to_sub_folders= '/home/malrabei/Downloads/paris/' # specify the directory to where you downloaded the paris dataset after unzipping the two folders as explained in the README file

direc_to_not_removed_file = "./not_removed.txt"
direc_to_query = './training_set/query/'
direc_to_positive = './training_set/positive/'
direc_to_negative = './training_set/negative/'
sub_direc = os.listdir(direc_to_sub_folders)

if not os.path.isdir('./training_set/'):
    os.mkdir('./training_set/')
if not os.path.isdir(direc_to_query):
    os.mkdir(direc_to_query)
if not os.path.isdir(direc_to_positive):
    os.mkdir(direc_to_positive)
if not os.path.isdir(direc_to_negative):
    os.mkdir((direc_to_negative))
sub_direc.remove('general')
""" the following code will delete the images that are marked as junk by the dataset authors """
removing_list = sub_direc
# removing_list.__delitem__(removing_list.index("general"))
for i in removing_list:
    file = open('./paris_120310/' + i + '_1_junk.txt')
    to_be_deleted = file.readlines()
    for j in range(len(to_be_deleted)):
        temp = to_be_deleted[j].strip('\n')
        try:
            os.remove(direc_to_sub_folders + i + '/' + temp + '.jpg')
            print("file: " + temp + ".jpg removed")
        except:
            print("skipping")

not_removed_files = open(direc_to_not_removed_file)
t = not_removed_files.readlines()
for i in removing_list:
    for j in range(len(t)):
        temp = t[j].strip('\n')
        try:
            os.remove(direc_to_sub_folders + i + '/' + temp + '.jpg')
            print("file: " + temp + ".jpg removed")
        except:
            print("skipping")

''' getting all the files in all subfolders '''
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles
all_files = getListOfFiles(direc_to_sub_folders)
dictOfLists = {}
query_list = all_files


for i in sub_direc: # getting the images names of all sub-directories
    dictOfLists[i] = getListOfFiles(direc_to_sub_folders+i)

for i in range(len(all_files)): # starting to divide all files into 3 folders: negative, positive, and query folders
    check = bool(dictOfLists)
    if not check:
        break
    randlist_name = random.choice(list(dictOfLists)) # choosing a random list

    if bool(dictOfLists[randlist_name]) & check:
        randlist = dictOfLists[randlist_name] # getting the files inside the chosen random list
        selected_img = randlist[0] # chose the first image from the chosen random list to be in the query folder
        dictOfLists[randlist_name].__delitem__(0) # remove this image from the list, so we avoid duplicates
        if  dictOfLists[randlist_name]: # making sure the list is not empty after removing the query image
            for j in range(10):

                rand_img = random.choice(randlist)
                shutil.copy(rand_img, direc_to_positive + 'img' + str(i) + '-' + str(j)+'-p.jpg')
                neg_img = random.choice(all_files)
                shutil.copy(selected_img, direc_to_query + 'img' + str(i)+ '-' + str(j)+ '-q.jpg')
                shutil.copy(neg_img, direc_to_negative + 'img' + str(i)  + '-' + str(j)+ '-n.jpg')
    elif not bool(dictOfLists[randlist_name]) & check:
        del dictOfLists[randlist_name]
    else:
        break
#

#
#
#
#
#
#
