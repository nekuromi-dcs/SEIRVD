import numpy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def SEIRVD_Model(t,y,N,lambda_birth,mu,baseline_nu,baseline_beta,sigma,gamma,alpha,epsilon,delta):
    V, S, E, I, R, D = y

    if t <= 100:
        nu = 0.0
    elif t<=130:
        nu = baseline_nu * (t-100) / 30.0
    else:
        nu = baseline_nu

    if I>(0.02*N):
        beta = 0.12
    elif I>(0.005*N):
        beta = 0.18
    else:
        beta = baseline_beta
            
    
    dV_dt = (nu*S) - ((sigma*beta*V*I)/N) - (mu*V)
    dS_dt = lambda_birth + delta*R - (beta*S*I)/N - nu*S - mu*S
    dE_dt = (beta*S*I)/N + (sigma*beta*I*V)/N - gamma*E - mu*E
    dI_dt = gamma*E - I*(mu+epsilon+alpha)
    dR_dt = alpha*I - mu*R - delta*R
    dD_dt = epsilon*I
    return [dV_dt, dS_dt,dE_dt,dI_dt,dR_dt,dD_dt]

#now that the SEIRVD model is ready, we need to initialize values of the things to put in it
N = 330000000        # Scale factor for density-dependent interactions
I0 = 1000           # Initial infectious individuals (100 active cases)
E0 = 2500          # Initial exposed individuals (incubation stage)
V0 = 0             # Initial vaccinated individuals
R0 = 0             # Initial recovered individuals
D0 = 0             # Initial deceased individuals
S0 = N - (V0 + E0 + I0 + R0 + D0)  # Rest of the population starts as susceptible

y0 = [V0, S0, E0, I0, R0, D0]

mu = 0.00004       # Natural death rate (approx. 1.5% annually)
# lambda_birth represents absolute daily births. Let's assume a stable baseline: 
# births roughly balancing out natural deaths (mu * N)
lambda_birth = mu * N 

baseline_nu = 0.003         # Vaccination rate (0.5% of susceptible group daily)
beta = 0.35        # Transmission rate parameter (average contacts times transmission probability)
sigma = 0.25       # Breakthrough infection multiplier (vaccinated are 80% protected, susceptibility = 20%)
gamma = 1 / 3.0    # Progress rate E -> I (5 days average incubation period)
alpha = 1 / 10.0   # Recovery rate I -> R (10 days average infectious period)
epsilon = 0.0005    # Disease-induced mortality rate (rate at which infected individuals pass away)

t_start = 0
t_end = 365*3
t_eval = numpy.linspace(t_start,t_end,10000)
#linspace creates a numpy array from 0 to 180 and it has 1000 spaces. like, 0.18, 0.36, 0.54, 0.72, 0.90...
#solve_ivp stands for Solve Initial Value Problem
solution = solve_ivp(
    fun=SEIRVD_Model,
    t_span=(t_start,t_end),
    y0=y0,
    t_eval=t_eval,
    args=(N,lambda_birth,mu,baseline_nu,beta,sigma,gamma,alpha,epsilon,delta:=0.0055),
    method='RK45'
    )
#solve_ivp returns something called "solution" which is a bundled object. It has the 2D Array for Solution Y, and the 1D Array for Solution T
V_out, S_out, E_out, I_out, R_out, D_out = solution.y
time_steps = solution.t
#creates a window that's 10inches wide and 6.5inches tall. why does it use inches???? weird
plt.figure(figsize=(10, 6.5))
plt.plot(time_steps, S_out, label='Susceptible (S)', color='#1f77b4', linewidth=2)
plt.plot(time_steps, V_out, label='Vaccinated (V)', color='#9467bd', linewidth=2)
plt.plot(time_steps, E_out, label='Exposed (E)', color='#ff7f0e', linewidth=2)
plt.plot(time_steps, I_out, label='Infectious (I)', color='#d62728', linewidth=2)
plt.plot(time_steps, R_out, label='Recovered (R)', color='#2ca02c', linewidth=2)
plt.plot(time_steps, D_out, label='Deceased (D)', color='#000000', linestyle='--', linewidth=1.5)

plt.title('VSEIRD Epidemiological Engine Simulation', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Timeline (Days)', fontsize=12)
plt.ylabel('Individual Count Headcount', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.show()

# --- 7. Post-Simulation Data Analytics Block ---

# Find the array index where the infectious population is at its absolute maximum
peak_index = numpy.argmax(I_out)

# Extract the exact peak value and the specific day it occurred
peak_cases = I_out[peak_index]
peak_day = time_steps[peak_index]

# Grab the final value from the deceased array at the end of the 180-day run
final_deaths = D_out[-1]

print("=" * 50)
print("             VSEIRD ENGINE SIMULATION REPORT            ")
print("=" * 50)
print(f"Total Base Population (N)   : {N:,} individuals")
print(f"Peak Infectious Headcount   : {int(peak_cases):,} active cases")
print(f"Outbreak Peak Velocity Day  : Day {peak_day:.1f}")
print(f"Final Deceased Toll ({int(t_end)}): {int(final_deaths):,} individuals")
print("=" * 50)
