// A simple linear program for optimizing profit gained
// Seeks to find optimal production of various metals within certain constraints
// Values and constraints are hard-coded into the program

using System; 
using System.Collections.Generic; 
using System.Linq; 
using System.Text; 
using System.Threading.Tasks;

namespace LP 
{ 
	class Program 
	{ 
		static void Main(string[] args) 
			{ 
				//Non-basic variables 
				int[] N = { 0, 1, 2, 3 };
				
				//Basic variables 	
				int[] B = { 4, 5, 6, 7, 8 };
				
				//Padded matrix 
				double[,] A = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, 
								{300.0, 30.0, 57.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{10000.0, 12000.0, 12300.0, 9100.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{500.0, 40.0, 63.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
								{40.0, 12.0, 57.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0} };
				
				//constraints 
				double[] b = { 0.0, 0.0, 0.0, 0.0, 2000.0, 1000.0, 1000000.0, 640.0, 432.0 };
				
				//coefficients of the objective function 
				double[] c = { 102, 844.6, 20.73, 853, 0.0, 0.0, 0.0, 0.0, 0.0 };
				
				//value we are maximizing 
				double v = 0.0;
				
				//initiate entering variable 
				int e = 0; 
				
				//initiate leaving variable 
				int l = -1;
				
				while (e != -1) 
				{ 
					bool foundCoefficient = false;
					//if there exists a variable in the non-basic list, then that variable becomes 
					//the entering variable
					
					
					foreach (int index in N) 
					{ 
						if (c[index] > 0) 
						{ 
							e = index; 
							foundCoefficient = true; 
							
							//Just find the first such positive coefficient 
							break; 
						} 
					} 
					
					if (!foundCoefficient) 
					{ 
						//This effectively breaks the loop 
						e = -1; 
					} 
					
					//Start the main loop of the Simplex Algorithm 
					else 
					{ 
						//Set to an extremely high number 
						double deltaMin = 99999999999999;
					
					//set leaving variable to -1 for this loop 
					l = -1;
					
					//Start the process of finding the minimum ratio for the current constraint 
					foreach (int i in B) 
					{ 
						//We only want to consider positive coefficients. 
						Console.WriteLine("i " + i + " e " + e); 
						if (A[i, e] > 0) 
						{ 
							//Find lowest check ratio so keep function within constraints 
							double delta = b[i] / A[i, e]; 
							
							if (delta < deltaMin) 
							{ 
								//Set to new min 
								deltaMin = delta; 
								
								//This will be the index we use if it is still the min 
								l = i; 
							} 
						} 
					} 
					
					//l is now the index of the smallest check ratio: deltaMin 
					if (deltaMin == 99999999999999) 
					{ 
						//Should not get here 
						Console.Out.WriteLine("Unbounded"); 
						Console.ReadKey(); 
					}
					
					else 
					{ 
						//Change values according to substitution of l 
						pivot(ref N, ref B, ref A, ref b, ref c, ref v, l, e); 
					}
				}
			}

			int n = b.Length; 
			double[] results = { 0, 0, 0, 0, 0, 0, 0, 0 }; 

			for (int i = 0; i < n; i++) 
			{ 
				if (B.Contains(i)) 
				{
					//the results will be the values of the basic variables 
					results[i] = b[i];
				} 
			}
			
			Console.WriteLine("copper " + results[0] + " gold " + results[1] + " silver " + results[2] + " platinum " + results[3] + " ? " + results[4] + " profit " + v); 
			Console.ReadKey();
		}
		
		static void pivot(ref int[] N, ref int[] B, ref double[,] A, ref double[] b, ref double[] c, ref double v, int l, int e) 
		{
			//Compute coefficients for the constraint for the new basic variable 
			x_e b[e] = b[l] / A[l, e]; 
			
			foreach (int j in N) 
			{ 
				if (j != e)
				{
					//Substituting new values for equality 
					A[e, j] = A[l, j] / A[l, e];
				} 
			}

			A[e, l] = 1 / A[l, e]; 
			
			//Compute the new coefficients for the other constraints 
			foreach (int i in B) 
			{ 
				if (i != l)
				{
					//Change values of b with substitution of l 
					b[i] = b[i] - A[i, e] * b[e];

					foreach (int j in N)
					{ 
						if (j != e)
						{
							//Change coefficients of the constrained equalities 
							A[i, j] = A[i, j] - A[i, e] * A[e, j];
						}
					}
					
					A[i, l] = -A[i, e] * A[e, l];
				} 
			}

			//Compute new objective function
			//New maximized value
			v = v + c[e] * b[e];
			
			foreach (int j in N)
			{ 
				if (j != e)
				{
					//Change coefficients according to new substitution of e 
					c[j] = c[j] - c[e] * A[e, j]; 
				}
			}

			//Change value of leaving variable. It will now be negative. 
			c[l] = -c[e] * A[e, l];
			
			int nIndex = -1; 
			
			foreach (int n in N)
			{ 
				nIndex += 1; 
				
				if (n == e)
				{ 
					break; 
				}
			} 
		
			//Replace entering variable with leaving variable in non-basic variables 
			N.SetValue(l, nIndex);

			int bIndex = -1; 
		
			foreach (int n in B)
			{ 
				bIndex += 1; 
			
				if (n == l)
				{ 
					break;
				} 
			} 
		
			//Replace leaving variable with entering variable in basic variables 
			B.SetValue(e, bIndex); 
			}	 
		}
	}
	
// Final solution:
// 27.55556 ounces gold
// 8.88888 ounces platinum
// Optimized profit: $19,218.93
