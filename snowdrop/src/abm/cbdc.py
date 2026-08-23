"""
Agent Based Model of Central Bank Digital Currency (CBDC).

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3959759

Requires mesa v2.3.2

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint,choice
from scipy.stats import poisson, truncnorm
from mesa import Agent, Model, batch_run
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

path = os.path.dirname(os.path.abspath(__file__))

def get_agents(model):
    agents = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agents)
    return x

retailers = []; commercial_banks = []
n_consumers=100; n_merchants=n_consumers/10 
n_commercial_banks=2; n_central_banks=1

consumer_card_credit_limit = 1.e3
consumer_cbdc_credit_limit = 1.e2
bank_limit = 1.e6

agents = None

class Consumers(Agent):
    """An agent has multiple options to exercise transactions.
    
       Consumers allocate their wealth and make purchase from Merchants.
       Consumers top-up CBDC wallets from bank deposits.
    """
    
    def __init__(self,model,unique_id,cash,cd,cbdc,
                 eta,p_mean,p_median,p_max,p_online,
                 wealth,wage,tau):
        """ 
            cash -     Amount of cash
            cd -       Value of credit card that consumer can charge
            cbdc -     Amount of CBDC wallet
            eta -      Average number of daily purchases
            p_mean -   Mean purchase value
            p_median - Median purchase value
            p_max -    Maximum purchase value
            p_online - The proportion of online purchases
            wealth -   Initial wealth distribution
            wage -     Labor wage
            tau -      CBDC top-up and cash withdrawal horizon
        """
        super().__init__(unique_id, model)
        self.cash = cash
        self.cd = cd
        self.cbdc = cbdc
        self.eta = eta
        self.p_mean = p_mean
        self.p_median = p_median
        self.p_max = p_max
        self.p_online = p_online
        self.wealth = wealth
        self.wage = wage
        self.tau = tau
      
    def purchases(self):
        """Transactions are represented by a bipartite network of consumers and retailers."""
        global agents
        
        cd = self.cd; cbdc = self.cbdc
        # Draw a total number of purchases
        n_purchases = poisson.rvs(mu=self.eta,size=1)[0]
        # Draw purchase size
        price = truncnorm.rvs(a=0,b=self.p_max,loc=self.p_mean,scale=1,size=1)[0]
        purchase_amount = n_purchases * price
        # Limit purchase amount
        purchase_amount = min(purchase_amount,self.wealth)
        # Draw a random merchant from the pool
        for i in range(1000):
            j = randint(n_consumers,n_consumers+n_merchants-1)
            merchant = agents[j]
            if isinstance(merchant,Merchants):
                break
   
        # Draw a random commercial bank from the pool 
        for i in range(1000):
            j = randint(n_consumers+n_merchants,n_consumers+n_merchants+n_commercial_banks-1)
            commercial_bank = agents[j]
            if isinstance(commercial_bank,CommercialBanks):
                break
        # Draw a random form of payment
        form_of_payment = randint(1,3)
        
        # Online payments (there is no cash payment)
        if form_of_payment == 2:
            amount = merchant.p_cards * min(self.tau*consumer_card_credit_limit,purchase_amount)
            diff = amount - self.cd 
            if diff > 0:
                amt = commercial_bank.withdraw(diff)
                self.cd = 0
            else:
                amt = amount
                self.cd = max(0, self.cd - amount)
        elif form_of_payment == 3:
            amount = merchant.p_cbdc * min(self.tau*consumer_cbdc_credit_limit,purchase_amount)
            diff = amount - self.cbdc 
            if diff > 0:
                amt = commercial_bank.issue_CBDC(diff)
                self.cbdc = 0
            else:
                amt = amount
                self.cbdc = max(0, self.cbdc - amount)
        
        # Offline payments (there is cash payment)
        if form_of_payment == 1:
            diff =  purchase_amount - self.cash
            if diff > 0:
                amt = commercial_bank.withdraw(diff)
                self.cash = 0
            else:  
                amt = purchase_amount  
                self.cash = max(0, self.cash - purchase_amount)
        elif form_of_payment == 2:
            amount = merchant.p_cards * min(self.tau*consumer_card_credit_limit,purchase_amount)
            diff = amount - self.cd 
            if diff > 0:
                amt = commercial_bank.withdraw(diff)
                self.cd = 0
            else:
                amt = amount
                self.cd = max(0, self.cd - amount)
        elif form_of_payment == 3:
            amount = merchant.p_cbdc * min(self.tau*consumer_cbdc_credit_limit,purchase_amount)
            diff = amount - self.cbdc 
            if diff > 0:
                amt = commercial_bank.issue_CBDC(diff)
                self.cbdc = 0
            else:
                amt = amount
                self.cbdc = max(0, self.cbdc - amount)
        
        if form_of_payment == 2 and not (merchant.accepts_cd or merchant.accepts_cd_and_cbdc):
                merchant.wealth -= merchant.loss * (cd-self.cd)
        elif form_of_payment == 3 and not(merchant.accepts_cbdc or merchant.accepts_cd_and_cbdc):
            merchant.wealth -= merchant.loss * (cbdc-self.cbdc)
        
        # Total payment
        total = amt
        
        # Wealth is allocated between cash, CBDC, and deposits used for payments via credit cards
        self.wealth = self.cash + self.cd + self.cbdc + 0*self.wage
        merchant.cash += self.wage
        
        self.cash += 1/(1+merchant.p_cards+merchant.p_cbdc) * self.wage
        self.cd += merchant.p_cards/(1+merchant.p_cards+merchant.p_cbdc) * self.wage
        self.cbdc += merchant.p_cbdc/(1+merchant.p_cards+merchant.p_cbdc) * self.wage
        
        # Merchant charges percentage of transaction amount for her/his service
        charge = merchant.charge * total
        merchant.wealth += charge
        self.wealth -= charge 
        
        return True
        
    def adoption_function(self,w,y):
        """ Probability of consumer having CBDC wallet. """
        return 0.25*(w+y)
        
    def step(self):
        self.purchases()
        

class Merchants(Agent):
    """Merchants accept cash and fraction of them accept credit cards or CBDS.
    """    
    
    def __init__(self,model,unique_id,wealth,charge,p_cards,p_cbdc,loss):
        """
            p_cards - The proportion of merchants who accept credit cards pyments
            p_cbdc -  The proportion of merchants who accept CBDC payments
        """
        super().__init__(unique_id, model)
        self.wealth = wealth
        self.charge = charge
        self.p_cards = p_cards
        self.p_cbdc = p_cbdc
        self.loss = loss
        # Random process
        self.accepts_cd = choice([True, False])
        self.accepts_cbdc = choice([True, False])
        self.accepts_cd_and_cbdc = self.accepts_cd and self.accepts_cbdc 
        self.cash = -1; self.cd = -1; self.cbdc = -1
    
    def step(self):
        pass

class CommercialBanks(Agent):
    """Banks issue deposits and set interest rate.
       Banks adjust their balance sheet.
    """

    def __init__(self,model,unique_id,deposit,cbdc,rd):
        """
        deposit - Initial deposit
        cbdc -    Initial CBDC borrowing
        rd -      Deposit interest rate
        """
        super().__init__(unique_id, model)
        self.deposit = deposit
        self.cash = deposit
        self.cbdc = cbdc
        self.rd = rd
        self.cd = 0
        self.wealth = deposit + self.cd
                
    def issue_CBDC(self,amount):
        """ Issue CBDC to a customer to top-up her/his wallet. """
        cb = agents[-1]
        diff = amount - self.cbdc
        if diff> 0:
            amt = self.borrow_from_CB(diff)
        else:
            amt = amount
        self.cbdc -= amt
        self.wealth = self.deposit + self.cbdc 
        lvrg = self.leverage()
        constraint = cb.leverage_constraint(lvrg)
        if constraint > 0:
            print("Benk leverage ratio is violated")
        return amt
            
    def borrow_from_CB(self,amount):
        """ Borrow CBDC from Central Bank. """
        cb = agents[-1]
        amt = min(amount,cb.beta)
        self.wealth += amt
        self.cbdc += amt
        return amt
        
    def withdraw(self,amount):
        """Provide consumer's withdrawals."""
        cb = agents[-1]
        amt = min(amount,cb.beta_cash,self.wealth)
        self.deposit -= amt
        self.wealth = self.deposit + self.cbdc 
        lvrg = self.leverage()
        constraint = cb.leverage_constraint(lvrg)
        if constraint > 0:
            print("Benk leverage ratio is violated")
        return amt
        
    def leverage(self):
        """ Leverage is the ratio of debt to the value of net assets. """
        debt = self.cbdc + self.deposit
        assets = self.wealth 
        if assets > debt:
            leverage = debt/(assets-debt)
        else:
            leverage = 1.e6
        return leverage

    def step(self):
        pass

class CentralBanks(Agent):
    """Central Bank conducts monetary policy and issues CBDC based on commercial banks demand."""

    def __init__(self,model,unique_id,beta,beta_cash,gamma,rb):
        """
        beta -      Maximum allowed CBDC balance
        beta_cash - Maximum cash withdrawals
        gamma -     Maximum leverage ratio
        rb -        CBDC borrowing rate
        """
        super().__init__(unique_id, model)
        self.beta = beta
        self.beta_cash = beta_cash
        self.gamma = gamma
        self.rb = rb
        self.wealth = -1; self.cash = -1; self.cd = -1; self.cbdc = -1
        
    def leverage_constraint(self,leverage):
        """Tweak the commercial bank leverage constraints."""
        return self.gamma - leverage

    def step(self):
        pass
    

class Engine(Model):
    """Payment tansactions model with number of agents."""

    def __init__(self,
                 # Consumers parameters
                 cash,cd,consumer_cbdc,eta,p_mean,p_median,p_max,
                 p_online,consumer_wealth,wage,tau,
                 # Merchants parameters
                 merchant_wealth,charge,p_cards,p_cbdc,loss,
                 # Commercial bank parameters
                 deposit,cbdc,rd,
                 # Central Bank parameters
                 beta,beta_cash,gamma,rb,
                 # Common parameters
                 n_consumers,n_merchants,n_commercial_banks,n_central_banks=1):
        global agents
        
        self.n_consumers = n_consumers
        self.n_merchants = n_merchants
        self.n_commercial_banks = n_commercial_banks
        self.n_central_banks = n_central_banks
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(n_consumers,n_consumers,True)
        self.running = True

        # Create consumer agents
        n1,n2 = 0,self.n_consumers
        for i in range(n1,n2):
            a = Consumers(self,i,cash,cd,consumer_cbdc,eta,
                         p_mean,p_median,p_max,p_online,
                         consumer_wealth,wage,tau)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(n_consumers)
            y = self.random.randrange(n_consumers)
            self.grid.place_agent(a, (x, y))
            
        # Create merchants agents
        n1,n2 = self.n_consumers,self.n_consumers+self.n_merchants
        for i in range(n1,n2):
            a = Merchants(self,i,merchant_wealth,charge,p_cards,p_cbdc,loss)
            x = self.random.randrange(n_consumers)
            y = self.random.randrange(n_consumers)
            self.schedule.add(a)
            self.grid.place_agent(a, (x, y))
            
        # Create commercial banks agents
        n1,n2 = n2,n2+self.n_commercial_banks
        for i in range(n1,n2):
            a = CommercialBanks(self,i,deposit,cbdc,rd)
            self.schedule.add(a)
            
        # Create central bank agent
        n1,n2 = n2,n2+self.n_central_banks
        for i in range(n1,n2):
            a = CentralBanks(self,i,beta,beta_cash,gamma,rb)
            self.schedule.add(a)
            
        agents = self.schedule.agents
        
        self.datacollector = DataCollector(
            model_reporters={"Wealth": get_agents}, 
            agent_reporters={"Wealth": "wealth", "Cash": "cash", "CD": "cd", "CBDC": "cbdc"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':
    
    # Consumers parameters
    cash=1.e4; cd=1.e3; consumer_cbdc=1.e1; eta=10
    p_mean=100; p_median=100; p_max=250
    p_online=0.2; consumer_wealth=1.e3; 
    wage=3.e2; tau=3
    
    # Merchants parameters
    merchant_wealth=1.e5; charge=0.1/365
    p_cards=0.9; p_cbdc=0.1; loss=0.1/365
    
    # Commercial Bank parameters
    deposit=3.e6; cbdc=1.e5; rd=0.5/365
    
    # Central Bank parameters
    beta=1.e6; beta_cash=300; gamma=1/0.03; rb=0.2/365
                     
    # Common parameters
    horizon=101
    n_consumers=100; n_merchants=n_consumers//10 # every merchant for 100 consumers
    n_commercial_banks=2; n_central_banks=1
    
    # Create a model with 1012 agents.
    model = Engine(# Consumers parameters
                   cash=cash,cd=cd,consumer_cbdc=consumer_cbdc,eta=eta,
                   p_mean=p_mean,p_median=p_median,p_max=p_max,
                   p_online=p_online,consumer_wealth=consumer_wealth,
                   wage=wage,tau=tau,
                   # Merchants parameters
                   merchant_wealth=merchant_wealth,charge=charge,
                   p_cards=p_cards,p_cbdc=p_cbdc,loss=loss,
                   # Commercial bank parameters
                   deposit=deposit,cbdc=cbdc,rd=rd,
                   # Central Bank parameters
                   beta=beta,beta_cash=beta_cash,gamma=gamma,rb=rb,
                   # Common parameters
                   n_consumers=n_consumers,n_merchants=n_merchants,
                   n_commercial_banks=n_commercial_banks,
                   n_central_banks=n_central_banks)
    
    for i in range(horizon):
        model.step()
    
    # Visualize the number of agents residing in each cell
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, xy = cell
        x,y = xy
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count
    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()
    plt.show()
        
    # Display agent wealth data
    agent_wealth = model.datacollector.get_agent_vars_dataframe()
    print(agent_wealth.head())
    
    # Plot histogram of agent wealth at the horizon end
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,15))
    plt.rcParams.update({'font.size': 17}) # must set in top
    n1 = n_consumers; n2 = n1+n_merchants; n3 = n2+n_commercial_banks
    end_wealth = agent_wealth.xs(horizon-1, level="Step")["Wealth"]
    consumers_end_wealth = end_wealth[:n1]
    merchants_end_wealth = end_wealth[n1:n2]
    commercial_banks_end_wealth = end_wealth[n2:n3]
    consumers_end_wealth.plot(kind="hist",title="Consumers Wealth",grid=True,xlabel="Period",ylabel="Wealth",ax=axes[0])
    merchants_end_wealth.plot(kind="hist",title="Merchants Wealth",grid=True,xlabel="Period",ylabel="Wealth",ax=axes[1])
    if not os.path.exists(os.path.abspath(os.path.join(path,"../../../graphs/CBDC"))):
        os.mkdir(os.path.abspath(os.path.join(path,"../../../graphs/CBDC")))
    fig.savefig(os.path.abspath(os.path.join(path,"../../../graphs/CBDC/Wealth_distribution.png")))
    plt.show()
    
    # Plot histogram of agent payment instruments value at the horizon end
    fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(10,15))
    plt.rcParams.update({'font.size': 17}) # must set in top
    end_cash = agent_wealth.xs(horizon-1, level="Step")["Cash"][:n1]
    end_cd = agent_wealth.xs(horizon-1, level="Step")["CD"][:n1]
    end_cbdc = agent_wealth.xs(horizon-1, level="Step")["CBDC"][:n1]
    end_cash.plot(kind="hist",title="Cash",grid=True,xlabel=None,ax=axes[0])
    end_cd.plot(kind="hist",title="Credit Card",grid=True,xlabel=None,ax=axes[1])
    end_cbdc.plot(kind="hist",title="CBDC",grid=True,xlabel="Period",ylabel="Wealth",ax=axes[2])
    fig.savefig(os.path.abspath(os.path.abspath(os.path.join(path,"../../../graphs/CBDC/Instruments_distribution.png"))))
    plt.show()
    
    # Plot the wealth of a given agent:
    N = 5
    consumer_legend = []; merchant_legend = []
    fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(10,20))
    plt.rcParams.update({'font.size': 15}) # must set in top
    for i in range(N):
        j = randint(1,n1-1)
        consumer_legend.append(f"{1+j}")
        one_agent_wealth = agent_wealth.xs(j, level="AgentID")
        one_agent_wealth.Wealth.plot(title="Consumers Wealth",xlabel=None,grid=True,ax=axes[0])
    #plt.legend(consumer_legend)
    plt.xlim(0,horizon-1)
    for i in range(N):
        j = randint(n1,n2-1)
        merchant_legend.append(f"{1+j}")
        one_agent_wealth = agent_wealth.xs(j, level="AgentID")
        one_agent_wealth.Wealth.plot(title="Merchants Wealth",xlabel="Period",grid=True,x=axes[1])
    #plt.legend(merchant_legend)
    plt.xlim(0,horizon-1)
    fig.savefig(os.path.abspath(os.path.join(path,"../../../graphs/CBDC/Instruments.png")))
    plt.show()
    
    
    fig, axes = plt.subplots(nrows=3,ncols=1,figsize=(12,20))
    plt.rcParams.update({'font.size': 15}) # must set in top
    
    df = agent_wealth.xs(0,level="AgentID")
    for j in range(1,n_consumers):
        df += agent_wealth.xs(j,level="AgentID")
    df /= n_consumers
    
    df.Cash.plot(title="Cash",xlabel=None,grid=True,lw=2,ax=axes[0])
    df.CD.plot(title="Credit Card",xlabel=None,grid=True,lw=2,ax=axes[1])
    df.CBDC.plot(title="CBDC",xlabel=None,grid=True,lw=2,ax=axes[2])
    fig.savefig(os.path.abspath(os.path.abspath(os.path.join(path,"../../../graphs/CBDC/Payment_instruments.png"))))
    plt.show()
        
    # consumer_legend = []; rnd = []
    # for i in range(N):
    #     j = randint(1,n1-1)
    #     rnd.append(j)
    #     consumer_legend.append(f"{1+j}")
    #     one_agent_wealth = agent_wealth.xs(j, level="AgentID")
    #     one_agent_wealth.Cash.plot(title="Cash",xlabel=None,grid=True,ax=axes[0])
    # #plt.legend(consumer_legend)
    # plt.xlim(0,horizon-1)
    # for i in range(N):
    #     j = randint(1,n1-1)
    #     rnd.append(j)
    #     consumer_legend.append(f"{1+j}")
    #     one_agent_wealth = agent_wealth.xs(j, level="AgentID")
    #     one_agent_wealth.CD.plot(title="Credit Card",xlabel=None,grid=True,ax=axes[1])
    # plt.xlim(0,horizon-1)
    # for i in range(N):
    #     j = randint(1,n1-1)
    #     rnd.append(j)
    #     consumer_legend.append(f"{1+j}")
    #     one_agent_wealth = agent_wealth.xs(j, level="AgentID")
    #     one_agent_wealth.CBDC.plot(title="CBDC",xlabel=None,grid=True,ax=axes[2])
    # plt.xlim(0,horizon-1)
    # plt.show()
    
    # save the agent data (stored in the pandas agent_wealth object) to CSV
    # agent_wealth.to_csv("agent_data.csv")

    ### Batch Run
    params = { # Consumers parameters
               "cash": cash, "cd": cd, "consumer_cbdc": consumer_cbdc, "eta": eta,
               "p_mean": p_mean, "p_median": p_median, "p_max": p_max,
               "p_online": p_online, "consumer_wealth": consumer_wealth, 
               "wage":wage, "tau": tau,
               # Merchants parameters
               "merchant_wealth": merchant_wealth, "charge": charge,
               "p_cards": p_cards, "p_cbdc": np.arange(0.1,1.0,0.2), "loss": loss,
               # Commercial banks parameters
               "deposit": deposit, "cbdc": cbdc, "rd": rd,
               # Central Bank parameters
               "beta": beta, "beta_cash": beta_cash, "gamma": gamma, "rb": rb,
               # Common parameters
               "n_consumers": n_consumers, "n_merchants": n_merchants,
               "n_commercial_banks": n_commercial_banks,
               "n_central_banks": n_central_banks
               }
                   
    results = batch_run(Engine,parameters=params,iterations=5,max_steps=horizon,
                        number_processes=10,data_collection_period=1,
                        display_progress=True)

    results_df = pd.DataFrame(results)
    print(results_df.keys())
    
    # Filter our results to only contain the data of one agent 
    # at the half of horizon step of each episode
    df = results_df[(results_df.AgentID == 0) & (results_df.Step == horizon)]
    v = df.values
    for j in range(1,n_consumers):
        df1 = results_df[(results_df.AgentID == j) & (results_df.Step == horizon)]
        v += df1.values
    v /= n_consumers
    df = pd.DataFrame(data=v,columns=df.columns)
    
    p_values = df.p_cbdc.values
    wealth_values = df.Wealth.values
    
    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
    plt.scatter(p_values, wealth_values)
    plt.xlabel("Proportion of merchants which accept CBDC payments")
    plt.ylabel("Consumer Wealth")
    plt.grid(True)
    fig.savefig(os.path.abspath(os.path.abspath(os.path.join(path,"../../../graphs/CBDC/proportion.png"))))
    plt.show()
    
    # Display the agent’s wealth at each time step of one specific episode
    one_episode_wealth = results_df[(results_df.p_cbdc == p_cbdc) & (results_df.iteration == 2)]
    # Then, print the columns of interest of the filtered data frame
    print(
        one_episode_wealth.to_string(
            index=False,columns=["Step","AgentID","Wealth"], max_rows=25)
    )   