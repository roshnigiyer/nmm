import javax.swing.*;
import java.util.*;
import java.io.*;
import java.lang.*;

class Tuple {
  double prob;
  boolean label;
  int weight;
  public Tuple(double prob, boolean label, int weight) {
    this.prob = prob; this.label = label; this.weight = weight;
  }
}

public class Main {
  public static int N, E, K = 4;
  public static double sgdStepSize = 0.01;    // may be different for different data
  public static final int negDictSize = 100000000;  // 100M
  public static String posFilename;

  public static double paramA = 1.0, paramB = 2.0, paramC = -1.0;
  public static double rAlpha = 1.5;
  public static double rLimit = 10;
  public static boolean WEIGHTED = false;
  public static double thetaReg = 1e-6;
  public static double tolerance = 1e-6;
  public static double eps = 1e-8;
  public static int MAX_EDGES = (int)1e15;
  public static int numEpochs = (int)500;
  public static boolean verbose = true;
  public static double tp = 0.9;
  public static final int shuffleSeed = 42;
  public static String outPrefix = "./";

  public static double nsw;
  public static Random rand;

  public static String outFilenameR, outFilenameTheta, outFilename1DTheta;
  public static List<Integer> edgeSources, edgeTargets, weights;
  public static Map<String, Integer> map;
  public static Map<Integer, String> invMap;
  public static int[] negDict = new int[negDictSize];
  public static Map<Integer, Integer> outDegree;

  public static void init() {
    rand = new Random(200);
    map = new HashMap<String, Integer>();
    invMap = new HashMap<Integer, String>();
    edgeSources = new ArrayList<Integer>();
    edgeTargets = new ArrayList<Integer>();
    weights = new ArrayList<Integer>();
    outDegree = new HashMap<Integer, Integer>();
    outFilenameR = outPrefix + "/res_r.txt";
    outFilenameTheta = outPrefix + "/res_z.txt";
    outFilename1DTheta = outPrefix + "/res_theta.txt";
    System.out.printf("[Info]: results will be saved to %s and %s\n", outFilenameR, outFilenameTheta);

    try {
      N = FileParser.readCSVDict(posFilename, "", map, invMap);

      E = FileParser.readCSVGraph(posFilename, map, edgeSources, edgeTargets, weights, N, negDict, outDegree, WEIGHTED);

      // use readCSVGraphApprox when there are too many edges (i.e. E > 100M)
      //E = FileParser.readCSVGraphApprox(posFilename, map, edgeSources, edgeTargets, weights, N, negDict, outDegree, WEIGHTED);

      nsw = 5;
    } catch (IOException e) {
      e.printStackTrace();
      System.exit(0);
    } 

    if (verbose) {
      System.out.printf("[Info] Number of nodes = %d\n", N);
      System.out.printf("[Info] Number of edges (pos) = %d, edges (neg) = %d\n", E, edgeSources.size()-E);
      System.out.printf("[Info] Density = %g, Negative sample weight = %g\n", 1.0*E/N/(N-1), nsw);
    }
  }

  /** 
   * calculate objective function (average probability)
   */
  public static double calcObj(double[][] R, double[][] theta, double gamma, double A, double B,
                               double C, double D, List<Integer> allEdges) {
    double res = 0.0;
    double pos_p = 0.0, neg_p = 0.0;
    int pos_count = 0, neg_count = 0;
//    double Z = Math.pow(rLimit, rAlpha+1) / (rAlpha+1);

    for (int _e = 0; _e < 2*E*tp; _e++) {
      int e = allEdges.get(_e);
      int s = edgeSources.get(e), t = edgeTargets.get(e);   // s cites t
      double lw = 1.0;
      double dz = sphericalDist(theta[s], theta[t]);
      double dr = l2norm(scalarVecAdd(eps, R[s])) - l2norm(scalarVecAdd(eps, R[t]));

      double pij_z = 1 / (1 + Math.exp(Math.exp(A) * dz + B));
      double pij_r = (Math.exp(Math.exp(C) * dr + D)) / (1 + Math.exp(Math.exp(C) * dr + D));

      double yHat = gamma * pij_z + (1 - gamma) * pij_r;

//      double dTheta = l2dist(theta[s], theta[t]);
//      double dij = paramA * (R[s]-R[t]) * (1-1/(1+dTheta)) - paramB * dTheta + paramC; //f_ij EQ 10
//      double pij = logis(dij);

      // positive links
      if (e < E) {
//        double log_pij = Math.log(pij) * lw; // EQ. 13 positive link optimization
//        log_pij *= weights.get(e);
//        pos_count += weights.get(e);
//        pos_p += log_pij;
//        res += log_pij;

        double log_yHat = Math.log(yHat) * lw;
        log_yHat *= weights.get(e);
        pos_count += weights.get(e);
        pos_p += log_yHat;
        res += log_yHat;
      }
      // negative links
      else {
//        double log_pij = Math.log(1-pij) * lw; // EQ. 13 negative link optimization
//        log_pij *= weights.get(e);
//	    neg_count += weights.get(e);
//	    neg_p += log_pij;
//	    res += nsw * log_pij;

        double log_yHat = Math.log(1-yHat) * lw;
        log_yHat *= weights.get(e);
        neg_count += weights.get(e);
        neg_p += log_yHat;
        res += nsw * log_yHat;
      }
    }

    if (pos_count+neg_count != 0) {
        res = res / -1 * (pos_count + nsw * neg_count);
    }

    if (pos_count+neg_count != 0 && verbose) {
      System.out.printf("Aver of pos = %f\n", pos_p/pos_count);
      System.out.printf("Aver of neg = %f, (without nsw) = %f\n", neg_p/neg_count*nsw, neg_p/neg_count);
      System.out.printf("Before regularization: %f\n", res/(pos_count+nsw*neg_count));
    }

    for (int n = 0; n < N; n++) {
        res += sphericalDist(sphProj(R[n]), theta[n]);
    }

//    for (int n = 0; n < N; n++) {
//      res += Math.pow(R[n], rAlpha) / Z; // p(r) .. but not doing log p(r) as per EQ. 13?
//      for (int k = 0; k < K; k++) {
//	    res += 0.5 * thetaReg * theta[n][k] * theta[n][k]; // logp(z)? BUT, logp(z) = - log(sigma) - 0.5log(2pi) - z^2 / 2 sigma_z^2?
//      }
//    }

    if (pos_count+neg_count != 0) return res;
    else return -1;
  }

  public static double[] scalarVecAdd(double eps, double[] a){
    for (int i = 0; i < a.length; i++) {
      a[i] += eps;
    }
    return a;
  }

  public static long nextLong(Random rng, long n) {
   // error checking and 2^x checking removed for simplicity.
   long bits, val;
   do {
      bits = (rng.nextLong() << 1) >>> 1;
      val = bits % n;
   } while (bits-val+(n-1) < 0L);
   return val;
  }

  public static double sphericalDist(double[] theta_i, double[] theta_j) {
    if (l1norm(theta_i) > 1) {
      theta_i = vectorMultiply(1 / l1norm(theta_i), theta_i);
    }
    if (l1norm(theta_j) > 1) {
      theta_j = vectorMultiply(1 / l1norm(theta_j), theta_j);
    }
      return Math.acos(dotProd(theta_i, theta_j));
  }

  public static double dotProd(double x[], double y[]) {
      if (x.length != y.length)
          throw new RuntimeException("Arrays must be same size");
      double sum = 0;
      for (int i = 0; i < x.length; i++)
          sum += x[i] * y[i];
      return sum;
  }

    /**
     * Returns the L1 norm of a vector.
     *
     * @param a the vector.
     * @return the L1 norm of a vector.
     */
    public static double l1norm(double[] a) {
        double norm = 0;
        for (double v : a) {
            norm += Math.abs(v);
        }
        return norm;
    }

    /**
    * Returns the sum of squares within a specific range.
    *
    * @param a the array.
    * @param fromIndex the index of the first element (inclusive).
    * @param toIndex the index of the last element (exclusive).
    * @return the sum of squares.
    */
    public static double sumSq(double[] a, int fromIndex, int toIndex) {
      double sq = 0.0;
      for (int i = fromIndex; i < toIndex; i++) {
        sq += a[i] * a[i];
      }
      return sq;
    }


    /**
    * Returns the L2 norm of a vector.
    *
    * @param a the vector.
    * @return the L2 norm of a vector.
    */
    public static double l2norm(double[] a) {
      return Math.sqrt(sumSq(a, 0, a.length));
    }

    public static double l2norm_squared(double[] a) {
      return Math.sqrt(sumSq(a, 0, a.length));
    }

    public static double[] vectorMultiply(double s, double[] v) {
        double[] result = new double[v.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = s * v[i];
        }
        return result;
    }

    public static double[] sphProj(double[] a) {
        if (l1norm(a) > 1) {
            a = vectorMultiply(1/l1norm(a), a);
        }
        return a;
    }

  /* return sigmoid(x) */
  public static double logis(double x) {
    if (x < -20) return 1e-9;
    else {
      double v = 1.0 / (1.0 + Math.exp(-x));
      if (v > 1-1e-9) return 1-1e-9;
      else return v;
    }
  }

  
  /* update parameters for a single link */
  public static void update(double[][] R, double[][] theta, double gamma, double A, double B,
                            double C, double D, int i, int j, boolean label, double[][] grad, double[] gradParams) {
    /*
     * [output]
     *	  grad[0][0~(K-1)]: gradient w.r.t. theta_i
     *	  grad[0][K~2K-1]: gradient w.r.t. R_i
     *	  grad[1][0~(K-1)]: gradient w.r.t. theta_j
     *	  grad[1][K~2K-1]: gradient w.r.t. R_j
     */
    // may need to change grad[][] size to 2K for R updates...

    double lw = 1.0;
    double dz = sphericalDist(theta[i], theta[j]);
    double dr = l2norm(scalarVecAdd(eps, R[i])) - l2norm(scalarVecAdd(eps, R[j]));
//    double pij_z = 1 / (1 + Math.exp(Math.exp(A) * dz + B));
//    double pij_r = (Math.exp(Math.exp(C) * dr + D)) / (1 + Math.exp(Math.exp(C) * dr + D));


//    double dTheta = l2dist(theta[i], theta[j]);
//    double dij = paramA * (R[i]-R[j]) * (1-1/(1+dTheta)) - paramB * dTheta + paramC; // EQ 10 in paper
//    double pij = logis(dij); // EQ 7 in paper

    int yij = label ? 1 : 0;
    double useNsw = label ? 1 : nsw;

    // calculate the derivative of EQ.13 by
      // (1) deriv(EQ. 13) / deriv(z_i)
      // (2) deriv(EQ. 13) / deriv(r_i)

    for (int k = 0; k < 2*K; k++) {
      if (k < K) {
	// dTheta

//	double g = (yij-pij) * ( 2 * paramA * (R[i]-R[j]) * (theta[i][k]-theta[j][k]) / ((1+dTheta)*(1+dTheta))
//	  - 2 * paramB * (theta[i][k]-theta[j][k]) );

        if (label) { //positive sampling

          if (l1norm(theta[j]) > 1) {
            theta[j] = vectorMultiply(1/l1norm(theta[j]), theta[j]);
          }

          if (l1norm(theta[i]) > 1) {
            theta[i] = vectorMultiply(1/l1norm(theta[i]), theta[i]);
          }

          if (l1norm(R[i]) > 1) {
            R[i] = vectorMultiply(1/l1norm(R[i]), R[i]);
          }

          if (l1norm(R[j]) > 1) {
            R[j] = vectorMultiply(1/l1norm(R[j]), R[j]);
          }

          double v1 = Math.pow(theta[j][k], 2) * Math.pow(theta[i][k], 2);
          if (v1 > 1) {
            v1 = logis(v1);
          }

          double v2 = Math.pow(R[i][k], 2) * Math.pow(theta[i][k], 2);
          if (v2 > 1) {
            v2 = logis(v2);
          }


          double g_theta_i = gamma * (A * theta[j][k] * Math.exp(A * dz + B)) /
                  (Math.sqrt(1 - Math.pow(theta[j][k], 2) * Math.pow(theta[i][k], 2)) * (Math.exp(A * dz + B) + 1))
                  + (-1 * R[i][k]) / (Math.sqrt(1 - Math.pow(R[i][k], 2) * Math.pow(theta[i][k], 2)));

          double g_theta_j = gamma * (A * theta[i][k] * Math.exp(A * dz + B)) /
                  (Math.sqrt(1 - Math.pow(theta[i][k], 2) * Math.pow(theta[j][k], 2)) * (Math.exp(A * dz + B) + 1))
                  + (-1 * R[j][k]) / (Math.sqrt(1 - Math.pow(R[j][k], 2) * Math.pow(theta[j][k], 2)));

          grad[0][k] += g_theta_i * lw * useNsw;
          grad[1][k] += g_theta_j * lw * useNsw;

          if (Double.isNaN(grad[0][k])) {
            System.out.println("line 306");
          }

          if (Double.isNaN(grad[1][k])) {
            System.out.println("line 309");
          }


//	grad[0][k] += g * lw * useNsw; // deriv(logsigmoid(p_ij)) / deriv(z_i) .. to do: change this to automatically compute the gradient of loss function
//	grad[1][k] -= g * lw * useNsw; // deriv(logsigmoid(p_ij)) / deriv(z_j) .. to do: change this to automatically compute the gradient of loss function

        } else { // negative sampling

          if (l1norm(theta[j]) > 1) {
            theta[j] = vectorMultiply(1/l1norm(theta[j]), theta[j]);
          }

          if (l1norm(theta[i]) > 1) {
            theta[i] = vectorMultiply(1/l1norm(theta[i]), theta[i]);
          }

          if (l1norm(R[i]) > 1) {
            R[i] = vectorMultiply(1/l1norm(R[i]), R[i]);
          }

          if (l1norm(R[j]) > 1) {
            R[j] = vectorMultiply(1/l1norm(R[j]), R[j]);
          }

          double v1 = Math.pow(theta[j][k], 2) * Math.pow(theta[i][k], 2);
          double v2 = Math.pow(R[i][k], 2) * Math.pow(theta[i][k], 2);
          double v3 = Math.pow(R[j][k], 2) * Math.pow(theta[j][k], 2);

          if (v1 > 1){
            v1 = logis(v1);
          }

          if (v2 > 1) {
            v2 = logis(v2);
          }

          if (v3 > 1) {
            v3 = logis(v3);
          }

          double g_theta_i = -1 * gamma * (A * theta[j][k] * Math.exp(A * dz + B)) /
                  (Math.sqrt(1 - Math.pow(theta[j][k], 2) * Math.pow(theta[i][k], 2)) * (Math.exp(A * dz + B) + 1) * (Math.exp(A * dz + B) - gamma + 1))
                  + (-1 * R[i][k]) / (Math.sqrt(1 - Math.pow(R[i][k], 2) * Math.pow(theta[i][k], 2)));

          double g_theta_j = -1 * gamma * (A * theta[i][k] * Math.exp(A * dz + B)) /
                  (Math.sqrt(1 - Math.pow(theta[i][k], 2) * Math.pow(theta[j][k], 2)) * (Math.exp(A * dz + B) + 1) * (Math.exp(A * dz + B) - gamma + 1))
                  + (-1 * R[j][k]) / (Math.sqrt(1 - Math.pow(R[j][k], 2) * Math.pow(theta[j][k], 2)));

          grad[0][k] += g_theta_i * lw * useNsw;
          grad[1][k] += g_theta_j * lw * useNsw;

          if (Double.isNaN(grad[0][k])) {
            System.out.println("line 331");
          }

          if (Double.isNaN(grad[1][k])) {
            System.out.println("line 335");
          }
        }


//	if (label)
//	  grad[0][k] -= thetaReg * theta[i][k] / outDegree.get(i) * nsw / (1+nsw);	// Gaussian prior gradient w/ neg. sampling? deriv(logp(z_i)) / deriv(z_i)
//	else
//	  grad[0][k] -= thetaReg * theta[i][k] / outDegree.get(i) * 1.0 / (1+nsw);	// Gaussian prior gradient w/ neg. sampling? deriv(logp(z_i)) / deriv(z_i)
      } else {
          // dR
          if (label) { // positive sampling
            double g_r_i = C / (logis(Math.exp(C * dr + D)) + 1);
            double g_r_j = -C / (logis(Math.exp(C * dr + D)) + 1);

            grad[0][k] += g_r_i * lw * useNsw / outDegree.get(i);
            grad[1][k] += g_r_j * lw * useNsw / outDegree.get(j);

            if (Double.isNaN(grad[0][k])) {
              System.out.println("line 353");
            }

            if (Double.isNaN(grad[1][k])) {
              System.out.println("line 358");
            }

          } else { // negative sampling
            double g_r_i = (C * (gamma - 1) * logis(Math.exp(C*dr+D))) / ((logis(Math.exp(C*dr+D))+1) * (gamma * logis(Math.exp(C*dr+D))+1));
            double g_r_j = (-1 * C * (gamma - 1) * logis(Math.exp(C*dr+D))) / ((logis(Math.exp(C*dr+D))+1) * (gamma * logis(Math.exp(C*dr+D))+1));

            grad[0][k] += g_r_i * lw * useNsw / outDegree.get(i);
            grad[1][k] += g_r_j * lw * useNsw / outDegree.get(j);

            if (Double.isNaN(grad[0][k])) {
              System.out.println("line 369");
            }

            if (Double.isNaN(grad[1][k])) {
              System.out.println("line 373");
            }
          }
//          if (!label) continue;
//          double g = (yij-pij) * paramA * (1-1/(1+dTheta));
//          grad[0][k] += g * lw * useNsw / outDegree.get(i); // deriv(logsigmoid(p_ij)) / deriv(r_i) .. to do: change this to automatically compute the gradient of loss function
//          grad[1][k] -= g * lw * useNsw / outDegree.get(i); // deriv(logsigmoid(p_ij)) / deriv(r_j) .. to do: change this to automatically compute the gradient of loss function

//          grad[0][k] += (R[i] != 0) ? (rAlpha / R[i]) / outDegree.get(i) : 100;	    // power law prior gradient? deriv(logp(r_i)) / deriv(r_i)
      }
    }
    // compute gradients for the params: gamma, A, B, C, D
    if (label) { // positive sampling
      double g_gamma = ( (1 / (1 + logis(Math.exp(C*dr+D)))) + (1 / (1+logis(Math.exp(A*dz+B)))) - 1   ) /
              ( (gamma / (logis(Math.exp(A*dz+B))+1)) + (1 - (1 / (logis(Math.exp(C*dr+D))+1)) ) * (1-gamma) );

      if (Double.isNaN(g_gamma)) {
        System.out.println("ps: g_gamma NaN");
      }

      double g_A = - (dz * logis(Math.exp(dz*A+B))) / ( logis(Math.exp(dz*A+B)) + 1);

      if (Double.isNaN(g_A)) {
        System.out.println("ps: g_A NaN");
      }

      double g_B = -1 * (logis(Math.exp(B+A*dz))) / (logis(Math.exp(B+A*dz))+1);

      if (Double.isNaN(g_B)) {
        System.out.println("ps: g_B NaN");
      }

      double g_C = dr / (logis(Math.exp(dr*C+D))+1);

      if (Double.isNaN(g_C)) {
        System.out.println("ps: g_C NaN");
      }

      double g_D = 1 / (logis(Math.exp(D + C*dr)) + 1);

      if (Double.isNaN(g_D)) {
        System.out.println("ps: g_D NaN");
      }

      gradParams[0] += g_gamma * lw * useNsw;
      gradParams[1] += g_A * lw * useNsw;
      gradParams[2] += g_B * lw * useNsw;
      gradParams[3] += g_C * lw * useNsw;
      gradParams[4] += g_D * lw * useNsw;



    } else { // negative sampling
      double g_gamma = (logis(Math.exp(C*dr+A*dz+D+B))-1) / ( gamma * (logis(Math.exp(C*dr+A*dz+D+B))) + (logis(Math.exp(A*dz+B))) + 1);

      if (Double.isNaN(g_gamma)) {
        System.out.println("ns: g_gamma NaN");
      }

      double g_A = (dz * gamma * logis(Math.exp(dz*A+B))) / ( (logis(Math.exp(dz*A+B)) + 1) * (logis(Math.exp(dz*A+B)) - gamma + 1) );

      if (Double.isNaN(g_A)) {
        System.out.println("ns: g_A NaN");
      }

      double g_B = (gamma * logis(Math.exp(B+A*dz))) / ((logis(Math.exp(B+A*dz))+1) * (logis(Math.exp(B+A*dz)) - gamma + 1));

      if (Double.isNaN(g_B)) {
        System.out.println("ns: g_B NaN");
      }

      double g_C = ( (gamma - 1) * dr * logis(Math.exp(dr*C+D))) / ( (logis(Math.exp(dr*C+D))+1) * (gamma * logis(Math.exp(dr*C+D))+1));

      if (Double.isNaN(g_C)) {
        System.out.println("ns: g_C NaN");
      }

      double g_D = ((gamma - 1) * logis(Math.exp(D+C*dr))) / ( (logis(Math.exp(D+C*dr)) + 1) * (gamma * logis(Math.exp(D+C*dr)) + 1) );

      if (Double.isNaN(g_D)) {
        System.out.println("ns: g_D NaN");
      }

      gradParams[0] += g_gamma * lw * useNsw;
      gradParams[1] += g_A * lw * useNsw;
      gradParams[2] += g_B * lw * useNsw;
      gradParams[3] += g_C * lw * useNsw;
      gradParams[4] += g_D * lw * useNsw;

    }
  }

  public static double l2dist(double[] vec1, double[] vec2) {
    double res = 0;
    int len = vec1.length;
    for (int k = 0; k < len; k++) res += (vec1[k]-vec2[k]) * (vec1[k]-vec2[k]);
    return res;
  }

  public static double l1dist(double[] vec1, double[] vec2) {
      double res = 0;
      int len = vec1.length;
      for (int k = 0; k < len; k++) res += (vec1[k]-vec2[k]);
      return res;
    }

   // function not being used
   public static double drCalcObj(List<Double> sim, double[] t, List<Integer> allEdges) {
    double res = 0;
    for (int _e = 0; _e < 2*E*tp; _e++) {
      int e = allEdges.get(_e);
      int i = edgeSources.get(e), j = edgeTargets.get(e);
      if (e < E) {
	double w = sim.get(e);
	res += w * Math.log(logis(CosSineTable.getCos(t[i]-t[j])));	  // likelihood
      } else {
	double w = sim.get(e);
	res += w * Math.log(1-logis(CosSineTable.getCos(t[i]-t[j])));
      }
    }
    res /= (2*E*tp);
    return res;
  }

  // function not being used
  public static void drUpdate(double w, double[] t, int i, int j, boolean label, double[] grad) {
    int y = label ? 1 : 0;
    double s = logis(CosSineTable.getCos(t[i] - t[j]));
    double g = -w * (y-s) * CosSineTable.getSine(t[i] - t[j]);    // weighted
    grad[0] += g; grad[1] -= g;
  }

  // function not being used
  public static void dimReduce(double[][] theta, double[] t) {
    List<Double> sim = new ArrayList<Double>();
    for (int e = 0; e < 2*E; e++) {
      int i = edgeSources.get(e), j = edgeTargets.get(e);
      double w = l2dist(theta[i], theta[j]);
      if (w < 0.5) 
	sim.add(1.0);
      else
	sim.add(0.0);
    }

    List<Integer> allEdges = new ArrayList<Integer>(2*E);     // index of all edges
    for (int i = 0; i < 2*E; i++) allEdges.add(i);
    Collections.shuffle(allEdges, new Random(shuffleSeed));
    double oldRes = drCalcObj(sim, t, allEdges), newRes = 0.0;

    long numEdges = 0;
    for (int _e = 0; _e < 2*E*tp; _e++) {
      int e = allEdges.get(_e);
      numEdges += weights.get(e);
    }
    int[][] edgeTable = new int[4][1<<30];
    long part = 0; int cur = 0;
    for (long i = 0; i < numEdges; i++) {
      if (i+1 > part) {
	part += weights.get(allEdges.get(cur));
	cur++;
      }
      int row = (int) (i >>> 30);
      int col = (int) (i & ((1 << 30) -1));
      edgeTable[row][col] = allEdges.get(cur-1);
    }

    for (int count = 0; count < 2*MAX_EDGES; count++) {
      if (count%(MAX_EDGES/10) == 0 && verbose) {
	System.out.printf("\n%d\n", count);
	newRes = drCalcObj(sim, t, allEdges);
	System.out.printf("[Train] obj = %f\n", newRes);
	test_performance(t);
      }
      long randl = nextLong(rand, numEdges);
      int row = (int) (randl >>> 30);
      int col = (int) (randl & ((1 << 30) - 1));
      int e = edgeTable[row][col];
      int i = edgeSources.get(e), j = edgeTargets.get(e);
      double w = sim.get(e);
      if (w == 0) continue;

      double[] grad = new double[2];
      if (e < E) 
	drUpdate(w, t, i, j, true, grad);
      else 
	drUpdate(0.5, t, i, j, false, grad);

      double thisStepSize = sgdStepSize * (1.0 - count / MAX_EDGES);
      t[i] += thisStepSize * grad[0];
      t[j] += thisStepSize * grad[1];
    }
  }

  /**
   * stochastic gradient descent
   */
  public static void runSGD(double[][] R, double[][] theta, double gamma, double A, double B,
                            double C, double D, double convergenceTol) {

    List<Integer> allEdges = new ArrayList<Integer>(2*E);     // index of all edges
    for (int i = 0; i < 2*E; i++) allEdges.add(i);
    Collections.shuffle(allEdges, new Random(shuffleSeed));
    double oldRes = calcObj(R, theta, gamma, A, B, C, D, allEdges), newRes = 0.0;
    //System.out.printf("[Train] obj (init) = %f\n", oldRes);

    long numEdges = 0;
    for (int _e = 0; _e < 2*E*tp; _e++) {
      int e = allEdges.get(_e);
      numEdges += weights.get(e);
    }
    if (verbose) System.out.printf("[Info] Number of edges in training, including multiplicity = %d\n", numEdges);
    int maxRow = 10000;
      int maxCol = 10000;
    //int[][] edgeTable = new int[4][1<<30]; NOTE: This caused a memory out of bounds error!
      int[][] edgeTable = new int[maxRow][maxCol];
    long part = 0; int cur = 0;
    for (long i = 0; i < numEdges; i++) {
      if (i+1 > part) {
	part += weights.get(allEdges.get(cur));
	cur++;
      }
      int row = (int) (i >>> 30);
      int col = (int) (i & ((1 << 30) -1));
      edgeTable[row][col] = allEdges.get(cur-1);
    }

    int ep = 1;
    for (int count = 0; count < MAX_EDGES; count++) {
      if (count%(MAX_EDGES/1000) == 0 && verbose) {
      System.out.println("Epoch:" + ep);
//	  System.out.printf("\n%d\n", ep);
      ep += 1;
	  newRes = calcObj(R, theta, gamma, A, B, C, D, allEdges);
	  System.out.printf("[Train] obj = %f\n", newRes);
	  test_performance(R, theta, gamma, A, B, C, D);
	  if (count/(MAX_EDGES/1000) >= numEpochs-1 && -(newRes-oldRes)/oldRes < convergenceTol) break;
	  oldRes = newRes;
	}

	long randl = nextLong(rand, numEdges);
	int row = (int) (randl >>> 30);
	int col = (int) (randl & ((1 << 30) - 1));
	int e = edgeTable[row][col];
	int s = edgeSources.get(e), t = edgeTargets.get(e);   // s -> t

	double[][] grad = new double[2][2*K];
    double[] gradParams = new double[5];
	if (e < E) {
	  update(R, theta, gamma, A, B, C, D, s, t,  true, grad, gradParams);
	} else {
	  update(R, theta, gamma, A, B, C, D, s, t, false, grad, gradParams);
	}

	// stepsize
        // TO DO: change this to descent not ascent e.g., theta[s][k] -= & R[s] -= and use RSGD
	double thisStepSize = sgdStepSize * (1.0 - count / MAX_EDGES);
    int index = 0;
	for (int k = 0; k < 2*K; k++) {
	  if (k < K) {
//	    theta[s][k] -= thisStepSize * grad[0][k];
//	    theta[t][k] -= thisStepSize * grad[1][k];
        theta[s][k] -= (-1 * thisStepSize) * (1 + (theta[s][k]*grad[0][k])/(l2norm(grad[0]))) * (1 - dotProd(theta[s], theta[s])) * grad[0][k] / l2norm(theta[s]);
        theta[t][k] -= (-1 * thisStepSize) * (1 + (theta[t][k]*grad[1][k])/(l2norm(grad[1]))) * (1 - dotProd(theta[t], theta[t])) * grad[1][k] / l2norm(theta[t]);

	  } else {
	    R[s][index] -= thisStepSize *  Math.pow( (1 - l2norm_squared(R[s]))/2,2) * grad[0][k];
	    R[t][index] -= thisStepSize * Math.pow( (1 - l2norm_squared(R[t]))/2,2) * grad[1][k];
        index += 1;
	  }
	}
//    gamma -= thisStepSize * gradParams[0];
//    gamma = logis(gamma);
    // kept getting different out of bounds error with the manually computed gradient e^(Adr+B) becomes inf for ex.
      // so making this hyperparam tuned for now.
//    A -= thisStepSize * gradParams[1];
//    B -= thisStepSize * gradParams[2];
//    C -= thisStepSize * gradParams[3];
//    D -= thisStepSize * gradParams[4];

	// projected gd
//	if (R[s] < 0) R[s] = 0;
//	if (R[t] < 0) R[t] = 0;
//	if (R[s] > rLimit) R[s] = rLimit;
//	if (R[t] > rLimit) R[t] = rLimit;
    }
  }

  // function not being used
  // 1d theta only
  public static void test_performance(double[] theta) {
    List<Integer> allEdges = new ArrayList<Integer>(2*E);	// index of all edges
    for (int i = 0; i < 2*E; i++) allEdges.add(i);
    Collections.shuffle(allEdges, new Random(shuffleSeed));

    double auc = 0.0, posCount = 0, negCount = 0;
    List<Tuple> res = new ArrayList<Tuple>();
    for (int _e = (int)(2*E*tp); _e < 2*E; _e++) {
      int e = allEdges.get(_e);
      int i = edgeSources.get(e), j = edgeTargets.get(e);	// s -> t
      int w = weights.get(e);
      double sij = CosSineTable.getCos(theta[i]-theta[j]);
      if (e < E) {
	res.add(new Tuple(sij, true, w));
	posCount += w;
      } else {
	res.add(new Tuple(sij, false, w));
	negCount += w;
      }
    }

    Collections.sort(res, new Comparator<Tuple>() {
      @Override
      public int compare(Tuple t1, Tuple t2) {
	if (t1.prob < t2.prob) return 1;
	else if (t1.prob > t2.prob) return -1;
	else return 0;
      }
    });

    double nx = 0.0, ox = 0.0, ny = 0.0;
    for (Tuple t: res) {
      if (t.label) {
	ny += 1.0 * t.weight / posCount;
      } else {
	nx += 1.0 * t.weight / negCount;
      }
      auc += ny * (nx-ox);
      ox = nx;
    }
    System.out.printf("[Test] AUC = %f, nx = %f, ny = %f\n", auc, nx, ny);
  }


  // k-dim R and K-dim theta (aka z)
  public static void test_performance(double[][] R, double[][] theta, double gamma, double A, double B, double C, double D) {
    List<Integer> allEdges = new ArrayList<Integer>(2*E);	// index of all edges
    for (int i = 0; i < 2*E; i++) allEdges.add(i);
    Collections.shuffle(allEdges, new Random(shuffleSeed));

    double auc = 0.0, posCount = 0, negCount = 0;
    List<Tuple> res = new ArrayList<Tuple>();
    for (int _e = (int)(2*E*tp); _e < 2*E; _e++) {
      int e = allEdges.get(_e);
      int s = edgeSources.get(e), t = edgeTargets.get(e);	// s -> t
      int w = weights.get(e);
      double dz = sphericalDist(theta[s], theta[t]);
//      double dTheta = l2dist(theta[s], theta[t]);
      double dr = l2norm(scalarVecAdd(eps, R[s])) - l2norm(scalarVecAdd(eps, R[t]));
      double pij_z = 1 / (1 + Math.exp(Math.exp(A) * dz + B));
      double pij_r = (Math.exp(Math.exp(C) * dr + D)) / (1 + Math.exp(Math.exp(C) * dr + D));
      double yHat = gamma * pij_z + (1 - gamma) * pij_r;

//      double dij = paramA * (R[s]-R[t]) * (1-1/(1+dTheta)) - paramB * dTheta + paramC;
      if (e < E) {
	    res.add(new Tuple(yHat, true, w));
	    posCount += w;
      } else {
	    res.add(new Tuple(yHat, false, w));
	    negCount += w;
      }
    }

    Collections.sort(res, new Comparator<Tuple>() {
      @Override
      public int compare(Tuple t1, Tuple t2) {
	if (t1.prob < t2.prob) return 1;
	else if (t1.prob > t2.prob) return -1;
	else return 0;
      }
    });

    double nx = 0.0, ox = 0.0, ny = 0.0;
    int lg = 0;
    for (Tuple t: res) {
      lg++;
      if (t.label) {
	ny += 1.0 * t.weight / posCount;
      } else {
	nx += 1.0 * t.weight / negCount;
      }
      auc += ny * (nx-ox);
      ox = nx;
    }
    System.out.printf("[Test] AUC = %f, nx = %f, ny = %f\n", auc, nx, ny);
  }

    public static double getRandomNumber(double minimum, double maximum)
    {
        Random random = new Random();
        return random.nextDouble() * (maximum - minimum) + minimum;
    }


  public static void start(String[] args) {
    init();
    if (verbose) System.out.println("[Info] Init done.\n");

    double[][] R = new double[N][K];
    double[][] theta = new double[N][K];
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
          R[n][k] = 1.0 * (rand.nextDouble() - 0.5);
          theta[n][k] = 1.0 * (rand.nextDouble() - 0.5);
      }
    }
    double gamma = 0.98; //but 1.0 shows bad performance...?

    double A = 4;
    double B = 2; // 2
    double C = 0.5;
    double D = 2; // 2
//    double A = getRandomNumber(0, 10);
//    double B = getRandomNumber(0, 10);
//    double C = getRandomNumber(0, 10);
//    double D = getRandomNumber(0, 10);


    runSGD(R, theta, gamma, A, B, C, D, tolerance);
    if (verbose) System.out.println("");

    try {
//      FileParser.output_2d_1(R, outFilenameR, invMap, false);
        FileParser.output_2d_2(R, outFilenameR, invMap);
        FileParser.output_2d_2(theta, outFilenameTheta, invMap);
    } catch (IOException e) {
      System.out.println("[I/O] File output error.");
    }

    /* 
     * the above scripts are enough to learn the embedding (r and z) in an
     * information network
     *
     * the scripts below are used for visualization (polarized coordinates)
     * uncomment if you want to see the polarized representation of data points
     */

    /*
    double[] d1t = new double[N];   // 1 dim theta
    for (int n = 0; n < N; n++) d1t[n] = rand.nextDouble() * 2 * Math.PI;
    dimReduce(theta, d1t);

    if (!verbose) test_performance(R, theta);
    if (!verbose) test_performance(d1t);

    try {
      FileParser.output_2d_1(d1t, outFilename1DTheta, invMap, true);
    } catch (IOException e) {
      System.out.println("[I/O] File output error.");
    }
    */
  }


  public static void main(String[] args) {
    try {
      ArgumentParser.parse(args);
    } catch (NumberFormatException e) {
      //e.printStackTrace();
      System.out.println("\nIllegal arguments.");
      ArgumentParser.help();
      System.exit(0);
    }

    long _start = System.currentTimeMillis();
    start(args);
    long _finish = System.currentTimeMillis();
    System.out.printf("Total time: %d seconds\n", (_finish-_start)/1000);
  }

}
