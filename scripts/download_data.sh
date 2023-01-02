data_folder="data"

# Download data from Kaggle
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge

# Unzip it
unzip jigsaw-toxic-comment-classification-challenge -d data

# Unzip the zipped!
zip_files=$(find $data_folder -name "*.zip")

# Loop through the zip files
for zip_file in $zip_files
do
  # Extract the zip file to the same directory
  unzip $zip_file -d $data_folder

  # Dispose inner zip file
  rm $zip_file
done

# Dispose main zip file
rm "jigsaw-toxic-comment-classification-challenge.zip"