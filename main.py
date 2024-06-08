"""
Name and Surname: Racha Badreddine
Student Number: 150210928
Date: 12/05/2024
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

#Checks the number of arguments given
if len(sys.argv) < 4:
    sys.exit("Too few arguments") 
elif len(sys.argv) > 4:
    sys.exit("Too many arguments")


################# SVD IMPLEMENTATION WITH QR METHOD  #####################
def QR_Factorization(Matrix):
    """
    Function that calculates the Q R Factorization of a given matrix usingGram Schmidt Method
    Parameters: Matrix A
    Returns: Orthogonal Matrix Q and upper triangular matrix R
    """
    rows, columns = Matrix.shape
    
    #Initializing Q and R with the right sizes
    Q = np.zeros((rows, columns))
    R = np.zeros((columns, columns))

    # Gram-Schmidt process
    for i in range(columns):
        #Every iteration a column of the matrix is taken
        v = Matrix[:, i]
        
        #Find the Orthogonal vector
        for j in range(i):
            v -= np.dot(v, Q[:, j]) * Q[:, j]
        #Make it Orthonormal
        Q[:, i] = v / np.linalg.norm(v)

    # Calculation of R, R= QTA
    for i in range(columns):
        for j in range(i, columns):
            R[i, j] = np.dot(Q[:, i], Matrix[:, j])

    return Q, R


#Calculating Eigen values and eigen vectors with QR factorization
def QR_Method(matrix, iterations=5000):
    """
    This function calculates the Eigen values and vectors for a given matrix
    Parameter: Matrix and number of iteration (Default 5000)
    
    Returns: array of eigen values and matrix of eigen vectors

    """    
    #Copy the matrix to work on it
    matrixk = np.copy(matrix)
    
    #Create an identity matrix to start with for the eigen vectors matrix
    row_number = matrixk.shape[0]
    eigen_vectors_matrix = np.eye(row_number)
    
    #The number of iterations i predefined here but may be changed and given as a parameter
    for k in range(iterations):
        
        #Take last element of the diagonal
        last = matrixk.item(row_number-1, row_number-1)
        #create a diagonal matrix out of this element
        diagonal_Last = last * np.eye(row_number)
        
        #Subtract it and perdform QR decomposition (BOTH linalg.qr and m)
        Q, R = np.linalg.qr(np.subtract(matrixk, diagonal_Last))
        #Q, R = QR_Factorization(np.subtract(matrixk, diagonal_Last))
        
        # we add last element again and calculate The natrixk
        matrixk = np.add(R @ Q, diagonal_Last)
        
        #Calculating the eigen vectors
        eigen_vectors_matrix = eigen_vectors_matrix @ Q
        
    # Get eigenvalues as the diagonal of the matrix
    eigenvalues = np.diag(matrixk)
    
    # Sorting eigenvalues and corresponding eigenvectors (Convention)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigen_vectors_matrix[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvectors

# SVD Decomposition A = U Sigma V.T
def SVD (Matrix):
    """
    Function to calculate the SVD decomposition of a given matrix
    Parameters: Matrix
    Returns: 3 Matrices such that Matrix = U S V.T 
    """
    #Calculate the transpose
    MatrixT = Matrix.T
    
    #Matrices we need to calculate eigenvectors
    ATA = np.dot(MatrixT, Matrix)
    
    #Gettin the eigen values and vectors by QR method
    eigen_values  , Vvectors = QR_Method(ATA)
    
    
    #Calculate singular values and put them into a diagonal matrix
    singular_values = np.sqrt(np.abs(eigen_values))
    # Sort eigenvalues based on their absolute values 
    singular_values_sorted = np.sort(singular_values)[::-1]
    Sigma = np.diag(np.sort(singular_values_sorted)[::-1])
    
    #Calculating Right left singular vectors U = A V Sigma.T
    Uvectors = Matrix @ Vvectors @ Sigma.T
    
    #Normalize U vectors
    column_magnitudes = np.linalg.norm(Uvectors, axis=0)
    Uvectors = Uvectors / column_magnitudes
    
    #Take the transpose of V
    Vvectors = Vvectors.T
    
    return Uvectors, Sigma, Vvectors

########################## Kabsch-Umeyama Algorithm ################
def read_create(file_path):
    """
    Function to open, read files and store data in a matrix
    #Returns a Matrix created out of file's lines
    #Parameters: file path
    """
    
    with open(file_path) as data_file :

        #To store data read from the file
        data_set = []

        #Get the lines of the file
        lines = data_file.readlines()

        #Store each line as a row  
        for line in lines:
            #Store each row in a list
            row = []
            
            #Read the values seperated by space
            for value in line.split(): 
                #Change strings to float numbers
                row.append(float(value))
                
            #Add each row to the data set
            data_set.append(row)

        #Create the matrix 
        matrix = np.array(data_set)
        
        return matrix
    
    
def Take_Correspondences (Matrix1, Matrix2, Correspondences):
    """
    #Function to build 2 matrices according to the correspondences
    #Return the two matrices with corresponding point
    #Parameters: 3 Matrices,mat1 , mat2 adnt he ones that stores the corrsponding row indexes
    """
    
    #Lists to store the new matrices
    new_mat1 = []
    new_mat2 = []
    
    #Read each row in Correspondences
    for row in Correspondences:
        #Row indexes in each row
        row1, row2 = row
        
        new_mat1.append(Matrix1[int(row1)])
        new_mat2.append(Matrix2[int(row2)])
    
    #Create Matrices out of the lists stored before
    new_mat1 = np.array(new_mat1)
    new_mat2 = np.array(new_mat2)
    
    
    return new_mat1, new_mat2


def rotation_calc(Q , P):
    """
    #Function To calculate The rotation matrix according to the Algorithm given
    #Returns the Rotation matrix
    #Parameters: Two matrices dxn
    """
    #Calculate the covariance matrix
    M = Q @ P.T
    
    #V, S , Wt = svd_simultaneous_power_iteration(M, 3)
    #Calculate the SVD decomposition 
    V, S , Wt = SVD(M)
    
    #Since W is used later
    W = Wt.T
    
    #size of diagonal matrix
    size = S.shape[0]
    
    #Build the diagonal matrix
    if (np.linalg.det(np.dot(V , W)) > 0):
        S = np.diag([1] * size )
    else:
        S = np.diag([1] * (size - 1) + [-1])
     
    #Calculate the Rotation matrix   
    R = W @ S @ V.T
    return R


def  KabschUmeyama(Q ,P ):
    """
    #Function To Apply Kabsch Umeyama on Two data sets
    #Returns the Rotation matrix and tranlation vector
    #Parameters: Two matrices nxd
    """
    #Both matrices should have the same size now
    assert Q.shape == P.shape
    
    # Calculating centroid of each data set
    centroid_Q = np.mean(Q, axis=0)
    centroid_P = np.mean(P, axis=0)
    
    #Center the points
    Q_centered = Q - centroid_Q
    P_centered = P - centroid_P
    
    #Calculating the rotation matrix (Transpose is sent to have nxd matrix) 
    rotation = rotation_calc(Q_centered.T, P_centered.T )
    
    #Calculating translation vector
    T = centroid_P - rotation @ centroid_Q
    
    return rotation, T.T 

def plot_3d_points(data, Title):
    """
    Function to plot a 3D data set in X, Y, Z axis
    #parameter: Matrix 
    """
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # Extract x, y, z coordinates from the matrix
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # Plot the points
    ax.scatter(xs, ys, zs)

    # Set labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.title(Title)

    # Show the plot
    plt.show()

def print_to_file(Rotation_matrix, Translation_vec, Merged):
    """
    Function to print Matrices to Files
    """
    
    rotation_path = "rotation_mat.txt"
    translation_path =  "translation_vec.txt"
    merged_path = "merged.txt"
    
    np.savetxt(rotation_path, Rotation_matrix, header='', comments='')
    np.savetxt(translation_path, Translation_vec, header='', comments='')
    np.savetxt(merged_path, Merged, header='', comments='')
    
# main function 
def main():
    #Read file paths from command line
    Mat1 = sys.argv[1]
    Mat2 = sys.argv[2]
    Correspondence = sys.argv[3]
    
    #Create the matrix of each file
    Original_Q = read_create(Mat1)
    Original_P = read_create(Mat2)
    Correspondence_matrix = read_create(Correspondence) 
    
    Q , P = Take_Correspondences(Original_Q, Original_P, Correspondence_matrix)
    
    #Apply algorithm and get rotation matrix and translation vector(returns a column vector)
    Rotation, translation_vec = KabschUmeyama(Q , P)
    
    #Finding Translatiom matrix To apply on P
    translation_vec = translation_vec.reshape((3, 1))
    Translation = np.tile(translation_vec, (1, Original_P.shape[0]))
    
    #Rotate and translate second data
    Rotated_translated_P = Rotation.T @ (Original_P.T - Translation)
   
    #Delete Overlapping data from second data set (Taking indices from second column of correspondences)
    list = [int(x) for x in Correspondence_matrix[:,1]]
    Rotated_translated_P = np.delete(Rotated_translated_P.T, list, axis=0)

    #Merge Data 
    merged_matrix = np.vstack((Original_Q, Rotated_translated_P))
    
    #Prints totation matrix, translation vector(as row vectr), and merged data
    print_to_file(Rotation, translation_vec.T, merged_matrix)
    
    plot_3d_points(merged_matrix, "Merged Data")
    
       

if __name__ == "__main__":
    main()

