from scipy.sparse         import lil_matrix    
from scipy.sparse.linalg  import spsolve, inv
from numpy.linalg         import solve, norm
from numpy.random         import rand
from scipy import sparse
import matplotlib.pyplot as plt

import numpy as np
from sys import getsizeof

import time
from datetime import datetime

def P_assign(n_1, r_1, P):
    res = 0
    for j in range(r_1 + 1):
        res += P[1][j] * P[n_1 - 1][r_1 - j]
    return res

def p_create(N, R, p, r):
    
    P = np.array([[0.0] * (R + 1) for i in range(N + 1)])
    
    P[0][0] = 1.0
    
    if R < r:
        P[1][R] = 1.0
        for i in range(R):
            P[1][i] = p[i]
            P[1][R] -= p[i]
    else:
        for i in range(r + 1):
            P[1][i] = p[i]                                                
    
    for n_1 in range(2, N + 1):
        for r_1 in range(R + 1):
            P[n_1][r_1] = P_assign(n_1, r_1, P)            
    
    return P

def p_mix(N, R, P_1, P_2, w_1, w_2):
    
    P = np.array([[0.0] * (R + 1) for i in range(N + 1)])
    
    P[0][0] = 1.0
    
    for i in range(R + 1):
        P[1][i] = P_1[1][i] * w_1 + P_2[1][i] * w_2
    
    for n_1 in range(2, N + 1):
        for r_1 in range(R + 1):
            P[n_1][r_1] = P_assign(n_1, r_1, P)           
            
    return P

def check_weight(w):
    
    w_f = w
    
    if w_f < 0.0:
        
        w_f = 0.0
        
    elif w_f > 1.0:
        
        w_f = 1.0
    
    return w_f

def release(j, n_1, r_1, P):
    result = (P[1][j] * P[n_1 - 1][r_1 - j]) / P[n_1][r_1]
    return result

def Psi(n_1, n_2, i, j, N, M, lamda, mu, gamma, alpha, v, demand_enough_1, demand_too_much_1, demand_enough_2, demand_too_much_2, release_same_3, release_demand_3):
    if i == j:
        if n_1 == N and n_2 == M:
            res = -(N * mu + N * gamma * (1.0 - release_same_3[N][i]) + M * alpha * (1.0 - v))
        elif n_1 == N:
            res = -(lamda * v + N * mu + N * gamma * (1.0 - release_same_3[N][i]) + n_2 * alpha * (1 - v))
        elif n_2 == M:
            res = -(lamda * demand_enough_1[i] + n_1 * mu + n_1 * gamma * (1.0 - release_same_3[n_1][i]) + M * alpha * (demand_enough_2[i] + (1 - v) * demand_too_much_2[i]))
        else:
            res = -(lamda * (demand_enough_1[i] + v * demand_too_much_1[i]) + n_1 * mu + n_1 * gamma * (1.0 - release_same_3[n_1][i]) + n_2 * alpha * (demand_enough_2[i] + (1 - v) * demand_too_much_2[i]))
    else:
        res = n_1 * gamma * release_demand_3[n_1][i][j]
    return res

def Mu(n_1, i, j, mu, gamma, v, P, demand_too_much_3):                                             
    res = n_1 * mu * release(i - j, n_1, i, P) + n_1 * gamma * release(i - j, n_1, i, P) * demand_too_much_3[j] * (1.0 - v)
    return res

def Mu_M(n_1, i, j, mu, gamma, P, demand_too_much_3):                                           
    res = n_1 * mu * release(i - j, n_1, i, P) + n_1 * gamma * release(i - j, n_1, i, P) * demand_too_much_3[j]
    return res

def Gamma(n_1, i, j, gamma, v, P, demand_too_much_3):                                              
    res = n_1 * gamma * release(i - j, n_1, i, P) * demand_too_much_3[j] * v
    return res
    
def analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_1, w_2, text_show):
    
    demand_enough_1 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1):
            result += P_1[1][j]
        demand_enough_1[r_1] = result
        
    demand_too_much_1 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1, R + 1):
            result += P_1[1][j]
        demand_too_much_1[r_1] = result
        
    demand_enough_2 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1):
            result += P_2[1][j]
        demand_enough_2[r_1] = result
        
    demand_too_much_2 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1, R + 1):
            result += P_2[1][j]
        demand_too_much_2[r_1] = result
        
    P_3 = p_mix(N, R, P_1, P_2, w_1, w_2)    
        
    demand_enough_3 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1):
            result += P_3[1][j]
        demand_enough_3[r_1] = result    
    
    demand_too_much_3 = np.zeros(R + 1)

    for r_1 in range(R + 1):
        result = 0.0
        for j in range(R - r_1 + 1, R + 1):
            result += P_3[1][j]
        demand_too_much_3[r_1] = result
        
    release_same_3 = np.array([[0.0] * (R + 1) for i in range(N + 1)])

    for n_1 in range(N + 1):
        for r_1 in range(R + 1):
            if P_3[n_1][r_1] > 0.0:
                result = 0.0
                for j in range(r_1 + 1):
                    result += release(j, n_1, r_1, P_3) * P_3[1][j]
                release_same_3[n_1][r_1] = result
            
    release_enough_3 = np.array([[0.0] * (R + 1) for i in range(N + 1)])

    for n_1 in range(N + 1):
        for r_1 in range(R + 1):
            if P_3[n_1][r_1] > 0.0:
                result = 0.0
                for j in range(r_1 + 1):
                    result += release(j, n_1, r_1, P_3) * demand_enough_3[r_1 - j]
                release_enough_3[n_1][r_1] = result
                
    release_too_much_3 = np.array([[0.0] * (R + 1) for i in range(N + 1)])

    for n_1 in range(N + 1):
        for r_1 in range(R + 1):
            if P_3[n_1][r_1] > 0.0:
                result = 0.0
                for j in range(r_1 + 1):
                    result += release(j, n_1, r_1, P_3) * demand_too_much_3[r_1 - j]
                release_too_much_3[n_1][r_1] = result 
    
    release_demand_3 = np.array([[[0.0] * (R + 1) for i in range(R + 1)] for j in range(N + 1)])

    for n_1 in range(N + 1):
        for i in range(R + 1):
            for j in range(R + 1):
                if P_3[n_1][i] > 0 and P_3[n_1][j] > 0:
                    if i < j:
                        for k in range(i + 1):
                            release_demand_3[n_1][i][j] += release(i - k, n_1, i, P_3) * P_3[1][j - k]
                    else:
                        for k in range(j + 1):
                            release_demand_3[n_1][i][j] += release(i - k, n_1, i, P_3) * P_3[1][j - k] 
    
    num   = np.array([[[-1.0] * (R + 1) for i in range(M + 1)] for j in range(N + 1)])
    shift = np.array([[[-1.0] * (R + 1) for i in range(M + 1)] for j in range(N + 1)])

    count_num   = 0
    shift_count = 0

    for n_2 in range(M + 1):
        num[0][n_2][0] = count_num
        shift[0][n_2][0] = 0
        count_num += 1

    for n_1 in range(1, N + 1):                                                           
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                if P_3[n_1][r_1] > 0.0:
                    num[n_1][n_2][r_1] = count_num
                    count_num += 1
                else:
                    shift_count += 1
                shift[n_1][n_2][r_1] = shift_count                           
                            
    A = lil_matrix((count_num, count_num))
    
    start_time = datetime.now()
    
    for n_2 in range(M + 1):                           
        A[int(n_2), int(n_2)] = -(lamda + n_2 * alpha)
        
    for n_2 in range(M + 1):
        for r_1 in range(R + 1):
            if num[1][n_2][r_1] != -1.0:
                x = shift[1][n_2][r_1]
                A[int(M + 1 + n_2 * (R + 1) + r_1 - x), int(n_2)] = mu
                A[int(n_2), int(M + 1 + n_2 * (R + 1) + r_1 - x)] = lamda * P_1[1][r_1]
                
    for n_2 in range(1, M + 1):
        for r_1 in range(R + 1):
            if num[1][n_2 - 1][r_1] != -1.0:
                x = shift[1][n_2 - 1][r_1]
                A[int(n_2), int(M + 1 + (n_2 - 1) * (R + 1) + r_1 - x)] = n_2 * alpha * P_2[1][r_1]
                
    for n_1 in range(1, N + 1):
        for n_2 in range(M + 1):
            for i in range(R + 1):
                for j in range(R + 1):
                    if num[n_1][n_2][i] != -1.0 and num[n_1][n_2][j] != -1.0:
                        x = shift[n_1][n_2][i]
                        y = shift[n_1][n_2][j]
                        A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + j - y)] = Psi(n_1, n_2, i, j, N, M, lamda, mu, gamma, alpha, v, demand_enough_1, demand_too_much_1, demand_enough_2, demand_too_much_2, release_same_3, release_demand_3)
         
    for n_1 in range(1, N):
        for n_2 in range(M):
            for i in range(R + 1):
                if num[n_1][n_2][i] != -1.0:
                    x = shift[n_1][n_2][i]
                    y = shift[n_1][n_2 + 1][i]
                    A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + R + 2 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - y)] = lamda * demand_too_much_1[i] * v
                       
    for n_2 in range(M):
        for i in range(R + 1):
            if num[N][n_2][i] != -1.0:
                x = shift[N][n_2][i]
                y = shift[N][n_2 + 1][i]                
                A[int(M + 1 + (N - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + R + 2 + (N - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - y)] = lamda * v
                
    for n_1 in range(1, N):
        for n_2 in range(1, M + 1):
            for i in range(R + 1):
                if num[n_1][n_2][i] != -1.0:
                    x = shift[n_1][n_2][i]
                    y = shift[n_1][n_2 - 1][i]
                    A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M - R + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - y)] = n_2 * alpha * demand_too_much_2[i] * (1.0 - v)   
                    
    for n_2 in range(1, M + 1):
        for i in range(R + 1):
            if num[N][n_2][i] != -1.0:
                x = shift[N][n_2][i]
                y = shift[N][n_2 - 1][i]
                A[int(M + 1 + (N - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M - R + (N - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - y)] = n_2 * alpha * (1.0 - v)                    
   
    for n_1 in range(1, N):
        for n_2 in range(M + 1):
            for i in range(R + 1):
                for j in range(i, R + 1):
                    if num[n_1][n_2][i] != -1.0 and num[n_1 + 1][n_2][j] != -1.0:
                        x = shift[n_1][n_2][i]
                        y = shift[n_1 + 1][n_2][j]
                        A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + 1 + n_1 * (M + 1) * (R + 1) + n_2 * (R + 1) + j - y)] = lamda * P_1[1][j - i]
                
    for n_1 in range(1, N):
        for n_2 in range(1, M + 1):
            for i in range(R + 1):
                for j in range(i, R + 1):
                    if num[n_1][n_2][i] != -1.0 and num[n_1 + 1][n_2 - 1][j] != -1.0:
                        x = shift[n_1][n_2][i]
                        y = shift[n_1 + 1][n_2 - 1][j]
                        A[int((M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x)), int(M - R + n_1 * (M + 1) * (R + 1) + n_2 * (R + 1) + j - y)] = n_2 * alpha * P_2[1][j - i]
                        
    for n_1 in range(2, N + 1):
        for n_2 in range(M):
            for i in range(R + 1):
                for j in range(i + 1):
                    if num[n_1][n_2][i] != -1.0 and num[n_1 - 1][n_2][j] != -1.0:
                        x = shift[n_1][n_2][i]
                        y = shift[n_1 - 1][n_2][j]
                        A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + 1 + (n_1 - 2) * (M + 1) * (R + 1) + n_2 * (R + 1) + j - y)] = Mu(n_1, i, j, mu, gamma, v, P_3, demand_too_much_3)
                
    for n_1 in range(2, N + 1):
        for i in range(R + 1):
            for j in range(i + 1):
                if num[n_1][M][i] != -1.0 and num[n_1 - 1][M][j] != -1.0:
                    x = shift[n_1][M][i]
                    y = shift[n_1 - 1][M][j]
                    A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + M * (R + 1) + i - x), int(M + 1 + (n_1 - 2) * (M + 1) * (R + 1) + M * (R + 1) + j - y)] = Mu_M(n_1, i, j, mu, gamma, P_3, demand_too_much_3) 
            
    for n_1 in range(2, N + 1):
        for n_2 in range(M):
            for i in range(R + 1):
                for j in range(i + 1):
                    if num[n_1][n_2][i] != -1.0 and num[n_1 - 1][n_2 + 1][j] != -1.0:
                        x = shift[n_1][n_2][i]
                        y = shift[n_1 - 1][n_2 + 1][j]
                        A[int(M + 1 + (n_1 - 1) * (M + 1) * (R + 1) + n_2 * (R + 1) + i - x), int(M + R + 2 + (n_1 - 2) * (M + 1) * (R + 1) + n_2 * (R + 1) + j - y)] = Gamma(n_1, i, j, gamma, v, P_3, demand_too_much_3)            
    
    t1 = datetime.now() - start_time    
        
    start_time = datetime.now()
    
    A_T = A.transpose()
    
    for j in range(count_num):
        A_T[count_num - 1, j] = 1.0    
    
    C = lil_matrix((count_num, 1))
    
    C[count_num - 1, 0] = 1.0
    
    A_T = A_T.tocsr()
    
    C   = C.tocsr()
    
    Q = spsolve(A_T, C)
    
    Q_sum = 0.0
    for i in range(count_num):
        Q_sum += Q[i]
    
    if text_show:
        
        print(Q_sum)
    
        print('')
    
        print(Q)
    
        print('')

    t2 = datetime.now() - start_time
    
    start_time = datetime.now()
    
    pi_1 = 0.0
    for n_1 in range(1, N):
        for n_2 in range(M):
            for r_1 in range(1, R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    pi_1 += (1.0 - v) * Q[int(k)] * demand_too_much_1[r_1]
        for r_1 in range(1, R + 1):
            k = num[n_1][M][r_1]
            if k != -1.0:
                pi_1 += Q[int(k)] * demand_too_much_1[r_1]        
            
    for n_2 in range(M):
        for r_1 in range(R + 1):
            k = num[N][n_2][r_1]
            if k != -1.0:
                pi_1 += (1.0 - v) * Q[int(k)]
    for r_1 in range(R + 1):
        k = num[N][M][r_1]
        if k != -1.0:
            pi_1 += Q[int(k)]
    
    if text_show:
        print('pi_1 = ', pi_1)  
    
    pi_2 = 0.0
    for n_1 in range(1, N):
        for n_2 in range(M):
            for r_1 in range(1, R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    pi_2 += v * Q[int(k)] * demand_too_much_1[r_1]
                    
    for n_2 in range(M):
        for r_1 in range(R + 1):
            k = num[N][n_2][r_1]
            if k != -1.0:
                pi_2 += v * Q[int(k)]
    
    if text_show:
        print('pi_2 = ', pi_2)
    
    pi_5 = 0.0
    for n_1 in range(N):
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    pi_5 += Q[int(k)] * demand_enough_1[r_1]
                    
    if text_show:
        
        print('pi_5 = ', pi_5)
        
        print('')
    
        print('pi_1 + pi_2 + pi_5 = ', pi_1 + pi_2 + pi_5)
    
        print('')
    
    orbit = np.array([[[-1.0] * (R + 1) for i in range(M + 1)] for j in range(N + 1)]) 
    count_orbit = 0
    sum_orbit = 0.0
    
    for n_1 in range(N + 1):                                                           
        for n_2 in range(1, M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    orbit[n_1][n_2][r_1] = Q[int(k)]
                    sum_orbit += Q[int(k)]
                    count_orbit += 1
    
    if text_show:
        print('sum_orbit = ', sum_orbit)
    
    sum_orbit_check = 0.0
    
    for n_1 in range(N + 1):                                                           
        for n_2 in range(1, M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    orbit[n_1][n_2][r_1] = orbit[n_1][n_2][r_1] * (1.0 / sum_orbit)
                    sum_orbit_check += orbit[n_1][n_2][r_1]
     
    if text_show:
        
        print('sum_orbit_check = ', sum_orbit_check)
    
        print('')
    
    if M > 0:
    
        orbit_tries = 0.0
        for n_1 in range(N + 1):                                                           
            for n_2 in range(1, M + 1):
                for r_1 in range(R + 1):
                    if P_3[n_1][r_1] > 0.0:
                        orbit_tries += orbit[n_1][n_2][r_1] * n_2
           
        orbit_back = 0.0
        for n_1 in range(1, N + 1):
            for n_2 in range(1, M + 1):
                for r_1 in range(R + 1):
                    if P_3[n_1][r_1] > 0.0:
                        if n_1 == N:
                            orbit_back += orbit[n_1][n_2][r_1] * n_2 * v
                        else:
                            orbit_back += orbit[n_1][n_2][r_1] * n_2 * v * demand_too_much_2[r_1]
                        
        pi_3 = orbit_back / orbit_tries
    
        if text_show:
            print('pi_3 = ', pi_3)
        
        orbit_lost = 0.0
        for n_1 in range(1, N + 1):                                                           
            for n_2 in range(1, M + 1):
                for r_1 in range(R + 1):
                    if P_3[n_1][r_1] > 0.0:
                        if n_1 == N:
                            orbit_lost += orbit[n_1][n_2][r_1] * n_2 * (1.0 - v)
                        else:
                            orbit_lost += orbit[n_1][n_2][r_1] * n_2 * (1.0 - v) * demand_too_much_2[r_1]
                        
        pi_4 = orbit_lost / orbit_tries
    
        if text_show:
            print('pi_4 = ', pi_4)
    
        orbit_taken = 0.0
        for n_1 in range(N):
            for n_2 in range(1, M + 1):
                for r_1 in range(R + 1):
                    if P_3[n_1][r_1] > 0.0:
                        orbit_taken += orbit[n_1][n_2][r_1] * n_2 * demand_enough_2[r_1]
              
        pi_6 = orbit_taken / orbit_tries        
       
        if text_show:
        
            print('pi_6 = ', pi_6)
        
            print('')
    
            print('pi_3 + pi_4 + pi_6 = ', pi_3 + pi_4 + pi_6)
    
            print('')
        
    else:
        pi_3 = 0.0
        pi_4 = 0.0
        pi_6 = 0.0
    
    pi_10 = pi_6 / (1.0 - pi_3)
    
    if text_show:
        print('pi_10 = ', pi_10)
    
    pi_11 = pi_4 / (1.0 - pi_3)
    
    if text_show:
        
        print('pi_11 = ', pi_11)
        
        print('')
    
        print('pi_10 + pi_11 = ', pi_10 + pi_11)
    
        print('')
    
    serv = np.array([[[-1.0] * (R + 1) for i in range(M + 1)] for j in range(N + 1)]) 
    count_serv = 0
    sum_serv = 0.0 
    
    for n_1 in range(1, N + 1):                                                           
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    serv[n_1][n_2][r_1] = Q[int(k)]
                    sum_serv += Q[int(k)]
                    count_serv += 1
    if text_show:                
        print('sum_serv = ', sum_serv)
    
    sum_serv_check = 0.0
    
    for n_1 in range(1, N + 1):                                                           
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    serv[n_1][n_2][r_1] = serv[n_1][n_2][r_1] * (1.0 / sum_serv)
                    sum_serv_check += serv[n_1][n_2][r_1]
    if text_show:
        
        print('sum_serv_check = ', sum_serv_check)
    
        print('')
    
    serv_signals = 0.0
    for n_1 in range(1, N + 1):
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                if P_3[n_1][r_1] > 0.0:
                    serv_signals += serv[n_1][n_2][r_1] * n_1
                    
    serv_orbit = 0.0
    for n_1 in range(2, N + 1):
        for n_2 in range(M):
            for r_1 in range(1, R + 1):
                if P_3[n_1][r_1] > 0.0:
                    serv_orbit += serv[n_1][n_2][r_1] * n_1 * release_too_much_3[n_1][r_1] * v
                    
    pi_7 = serv_orbit / serv_signals
    
    if text_show:
        print('pi_7 = ', pi_7)
    
    serv_lost = 0.0
    for n_1 in range(2, N + 1):
        for n_2 in range(M + 1):
            for r_1 in range(1, R + 1):
                if P_3[n_1][r_1] > 0.0:
                    if n_2 == M:
                        serv_lost += serv[n_1][n_2][r_1] * n_1 * release_too_much_3[n_1][r_1]
                    else:
                        serv_lost += serv[n_1][n_2][r_1] * n_1 * release_too_much_3[n_1][r_1] * (1.0 - v)
                        
    pi_8 = serv_lost / serv_signals                    
      
    if text_show:    
        print('pi_8 = ', pi_8)
    
    serv_con = 0.0
    for n_1 in range(1, N + 1):
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                if P_3[n_1][r_1] > 0.0:
                    serv_con += serv[n_1][n_2][r_1] * n_1 * release_enough_3[n_1][r_1]
                    
    pi_9 = serv_con / serv_signals
    
    if text_show:
        
        print('pi_9 = ', pi_9)
        
        print('')
    
        print('pi_7 + pi_8 + pi_9 = ', pi_7 + pi_8 + pi_9)
    
        print('')
    
    n_1_av = 0.0
    for n_1 in range(1, N + 1):
        for n_2 in range(M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    n_1_av += n_1 * Q[int(k)]
    
    if text_show:
        print('n_1_av = ', n_1_av)        
    
    n_2_av = 0.0  
    for n_1 in range(N + 1):
        for n_2 in range(1, M + 1):
            for r_1 in range(R + 1):
                k = num[n_1][n_2][r_1]
                if k != -1.0:
                    n_2_av += n_2 * Q[int(k)]
    
    if text_show:
        
        print('n_2_av = ', n_2_av)
                
        print('')
    
    pi_12 = (n_1_av * gamma * pi_7) / (lamda * pi_5 + n_2_av * alpha * pi_6)
    
    if text_show:
        print('pi_12 = ', pi_12)
    
    pi_13 = (n_1_av * gamma * pi_8) / (lamda * pi_5 + n_2_av * alpha * pi_6)
    
    if text_show:
        
        print('pi_13 = ', pi_13)
    
        print('')
        
    if text_show:
        
        print('1 - pi_12 - pi_13 = ', 1.0 - pi_12 - pi_13)
        
        print('')
        
    pi_14 = ((pi_5 * (1.0 - pi_12 - pi_13)) / (1.0 - pi_12 * pi_10)) + ((pi_2 * pi_10 * (1.0 - pi_12 - pi_13)) / (1.0 - pi_12 * pi_10))
    
    if text_show:
        print('pi_14 = ', pi_14)
        
    pi_15 = pi_1 + pi_2 * pi_11 + (pi_5 * pi_13) / (1.0 - pi_12 * pi_10) + (pi_5 * pi_12 * pi_11) / (1.0 - pi_10 * pi_12) + (pi_2 * pi_10 * pi_13) / (1.0 - pi_12 * pi_10) + (pi_2 * pi_10 * pi_12 * pi_11) / (1.0 - pi_10 * pi_12)
    
    if text_show:
        
        print('pi_15 = ', pi_15)
        
        print('')
        
        print('pi_14 + pi_15 = ', pi_14 + pi_15)
        
        print('')
        
    pi_16 = pi_5 * (1.0 - pi_12 - pi_13)
    
    if text_show:
        
        print('pi_16 = ', pi_16)
        
        print('pi_w  = ', pi_14 - pi_16)
        
        print('')
        
    pi_17 = 1.0 - pi_1 - pi_16 - pi_5 * pi_13
    
    if text_show:
        
        print('pi_17 = ', pi_17)
        
        print('')
                    
    b = 0.0
    for n_1 in range(N + 1):
        for n_2 in range(M + 1):
            for r_1 in range(1, R + 1):
                k = num[n_1][n_2][r_1]
                #print(k)
                if k != -1.0:
                    b += r_1 * Q[int(k)] 
    
    B = pi_1 + pi_2 * pi_11
    
    I = pi_13 / (1.0 - pi_12 * pi_10) + pi_12 * pi_11 / (1.0 - pi_10 * pi_12)
    
    I_2 = (1.0 - B) * I
    
    if text_show:
        
        print('I_2 = ', I_2)
        
        print('')
        
    if text_show:
        
        print('pi_15   = ', pi_15)
        
        print('B + I_2 = ', B + I_2)
        
        print('')
        
    S = pi_16
    
    W = pi_14 - pi_16
    
    O = pi_17
    
    b = b / R
    
    U = (b * I_2) / (I_2 + pi_14)
    
    t3 = datetime.now() - start_time
    
    if text_show:
    
        print('t1 = ', t1)
        
        print('t2 = ', t2)
        
        print('t3 = ', t3)
        
        print('')
                    
    return B, b, I_2, S, W, O, U, n_2_av, pi_1, pi_5, pi_6
    
def weight_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_1_in, w_2_in, prec, text_show):
    
    k = 1
    
    B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6 = analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_1_in, w_2_in, False)
    
    w_11 = (lamda * pi_5) / (lamda * pi_5 + alpha * n_2_av * pi_6)
    
    w_11 = check_weight(w_11)
    
    w_21 = (alpha * n_2_av * pi_6) / (lamda * pi_5 + alpha * n_2_av * pi_6)
    
    w_21 = check_weight(w_21)
    
    if text_show:
    
        print('k = 0')
    
        print('w_1 =', w_11, end = ', ')
        
        print('w_2 =', w_21)
        
        print('')
    
    w_12 = w_11
    
    w_22 = w_21    
    
    B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6 = analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_11, w_21, False)
    
    w_11 = (lamda * pi_5) / (lamda * pi_5 + alpha * n_2_av * pi_6)
    
    w_11 = check_weight(w_11)   
    
    w_21 = (alpha * n_2_av * pi_6) / (lamda * pi_5 + alpha * n_2_av * pi_6)
    
    w_21 = check_weight(w_21)    
    
    if text_show:
    
        print('k = 1')
    
        print('w_1 =', w_11, end = ', ')
        
        print('w_2 =', w_21)
        
        print('')
    
    while abs(w_11 - w_12) > prec or abs(w_21 - w_22) > prec:
        
        k += 1
        
        w_12 = w_11
    
        w_22 = w_21
        
        B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6 = analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_11, w_21, False)
    
        w_11 = (lamda * pi_5) / (lamda * pi_5 + alpha * n_2_av * pi_6)
        
        w_11 = check_weight(w_11)       
    
        w_21 = (alpha * n_2_av * pi_6) / (lamda * pi_5 + alpha * n_2_av * pi_6)
        
        w_21 = check_weight(w_21)       
        
        if text_show:
            
            print('k =', k)
        
            print('w_1 =', w_11, end = ', ')
        
            print('w_2 =', w_21)
            
            print('')    
        
    return w_11, w_21
    
def analytical_account(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, text_show, adapt_resource):
    
    w_1_in = 1.0
    
    w_2_in = 0.0
    
    prec = 0.001
    
    if adapt_resource:
    
        w_1, w_2 = weight_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_1_in, w_2_in, prec, text_show)
    
        B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6 = analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_2, w_1, w_2, text_show)
        
    else:
        
        B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6 = analytical_calculation(N, M, R, lamda, mu, gamma, alpha, v, P_1, P_1, w_1_in, w_2_in, text_show)
    
    return B, b, I, S, W, O, U, n_2_av, pi_1, pi_5, pi_6