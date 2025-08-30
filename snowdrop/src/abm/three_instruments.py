"""
Agent Based Model of payment instruments dynamics.

https://mesa.readthedocs.io/en/latest/tutorials/intro_tutorial.html

Requires mesa v2.3.2
"""
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from enum import unique,Enum
from random import randint,uniform
from mesa import Agent, Model, batch_run
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

path = os.path.dirname(os.path.abspath(__file__))

customers = {}

labels =  {
    ##### MERCHANTS
    "pc":    "merchant fixed cost of accepting cash",
    "qc":    "merchant variable cost of accepting cash",
    "pd":    "merchant fixed cost of accepting credit cards",
    "qd":    "merchant variable cost of accepting credit cards",
    "ps":    "merchant fixed cost of accepting CBDC",
    "qs":    "merchant variable cost of accepting CBDC",
    "ksi":   "cost of missed sale",
    
    ##### CUSTOMERS
    "cf":    "customer fixed cost of cash",
    "beta":  "customer cash_benefit",
    "cd":    "customer fixed cost of credit card",
    "rd":    "customer credit card_benefit",
    "cs":    "customer fixed cost of CBDC",
    "rs":    "customer CBDC benefit",
    
    ##### INSTRUMENTS
    "x":     "proportion of customers holding cash (c) only",
    "y":     "proportion of customers holding credit cards (d) only",
    "z":     "proportion of customers holding CBDC (cbc) only",
    "xy_x":  "proportion of customers who have two instruments (c and d) and pay by cash", 
    "xy_y":  "proportion of customers who have two instruments (c and d) and pay by credit card",
    "xz_c":  "proportion of customers who have two instruments (c and cbdc) and pay by cash",
    "xz_z":  "proportion of customers who have two instruments (c and cbdc) and pay by cbdc", 
    "yz_y":  "proportion of customers who have two instruments (cd and cbdc) and pay by credit card",
    "yz_z":  "proportion of customers who have two instruments (cd and cbdc) and pay by cbdc", 
    "xyz_x": "proportion of customers who have three instruments and pay by cash", 
    "xyz_d": "proportion of customers who have three instruments and pay by credit card", 
    "xyz_z": "proportion of customers who have three instruments and pay by cbdc"
    }

# Merchant positions
pos_x = {}; pos_y = {}; customers = {}
            
@unique
class States(Enum):
    CASH  = 1  
    CREDITCARD = 2  
    CBDC = 4  
    CASH_AND_CREDITCARD = 8  
    CASH_AND_CBDC = 16
    CREDITCARD_AND_CBDC = 32
    CASH_AND_CREDITCARD_AND_CBDC = 64
    
states = States._member_map_.keys()
n_states = len(States)

def get_states(rnd,n_states):
    """ Converts random integer number to random states."""
    states = [0]*n_states
    s = bin(rnd)
    s = s.replace("0b","")
    s = [int(x) for x in list(s)]
    
    if len(s) > 1+n_states:
        s = s[:1+n_states]
        
    if len(s) == 1+n_states:
        states = [1]*n_states
    elif len(s) == n_states:
        states[:3] = s[:3]
        if bool(s[n_states-1]):  
            states[1] = states[3] = 1
        if bool(s[n_states-2]):  
            states[0] = states[2] = 1
        if bool(s[n_states-3]):  
            states[0] = states[1] = 1
    elif len(s) == n_states-1:
        states[:3] = s[:3]
        if bool(s[-1]):  
            states[1] = states[3] = 1
        if bool(s[-2]):  
            states[0] = states[2] = 1
        if bool(s[-3]):  
            states[0] = states[1] = 1
    elif len(s) == n_states-2:
        states[:3] = s[:3]
        if bool(s[-1]):  states[1] = states[3] = 1
    elif len(s) <= n_states:
        states[:len(s)] = s
        
    return states
    
def compute_customer_statistics(model):
    """ Compute number of customers that hold cash, credit cards and cbdc. """
    agents = [a for a in model.schedule.agents if isinstance(a,Customer)]
    holds_CASH = [agent.holds_CASH for agent in agents]
    holds_CD = [agent.holds_CD for agent in agents]
    holds_CBDC = [agent.holds_CBDC for agent in agents]
    holds_CASH_CD = [agent.holds_CASH_CD for agent in agents]
    holds_CASH_CBDC = [agent.holds_CASH_CBDC for agent in agents]
    holds_CD_CBDC = [agent.holds_CD_CBDC for agent in agents]
    holds_CASH_CD_CBDC  = [agent.holds_CASH_CD_CBDC for agent in agents]
    total = sum(holds_CASH) + sum(holds_CBDC) + sum(holds_CASH_CD) + sum(holds_CASH_CD) \
          + sum(holds_CASH_CBDC) + sum(holds_CD_CBDC) + sum(holds_CASH_CD_CBDC)
    # Compute proportions      
    total = max(1.e-5,total)
    holds_CASH = sum(holds_CASH) / total
    holds_CD = sum(holds_CD) / total
    holds_CBDC = sum(holds_CBDC) / total
    holds_CASH_CD = sum(holds_CASH_CD) / total
    holds_CASH_CBDC = sum(holds_CASH_CBDC) / total
    holds_CD_CBDC = sum(holds_CD_CBDC) / total
    holds_CASH_CD_CBDC = sum(holds_CASH_CD_CBDC) / total
    
    proportions  = [holds_CASH,holds_CD,holds_CBDC,holds_CASH_CD,holds_CASH_CBDC,holds_CD_CBDC,holds_CASH_CD_CBDC]
    return proportions
    
def compute_merchant_statistics(model):
    """ Compute number of customers that hold cash, credit cards and cbdc. """
    agents = [a for a in model.schedule.agents if isinstance(a,Merchant)]
    accepts_CASH = [agent.accepts_CASH for agent in agents]
    accepts_CD = [agent.accepts_CD for agent in agents]
    accepts_CBDC = [agent.accepts_CBDC for agent in agents]
    accepts_CASH_CD = [agent.accepts_CASH_CD for agent in agents]
    accepts_CASH_CBDC = [agent.accepts_CASH_CBDC for agent in agents]
    accepts_CD_CBDC = [agent.accepts_CD_CBDC for agent in agents]
    accepts_CASH_CD_CBDC  = [agent.accepts_CASH_CD_CBDC for agent in agents]
    total = sum(accepts_CASH) + sum(accepts_CBDC) + sum(accepts_CASH_CD) + sum(accepts_CASH_CD) \
          + sum(accepts_CASH_CBDC) + sum(accepts_CD_CBDC) + sum(accepts_CASH_CD_CBDC)
    # Compute proportions      
    total = max(1.e-5,total)
    accepts_CASH = sum(accepts_CASH) / total
    accepts_CD = sum(accepts_CD) / total
    accepts_CBDC = sum(accepts_CBDC) / total
    accepts_CASH_CD = sum(accepts_CASH_CD) / total
    accepts_CASH_CBDC = sum(accepts_CASH_CBDC) / total
    accepts_CD_CBDC = sum(accepts_CD_CBDC) / total
    accepts_CASH_CD_CBDC = sum(accepts_CASH_CD_CBDC) / total
    
    proportions  = [accepts_CASH,accepts_CD,accepts_CBDC,accepts_CASH_CD,accepts_CASH_CBDC,accepts_CD_CBDC,accepts_CASH_CD_CBDC]
    return proportions

def get_closest_merchant(x0,y0):
    """ """
    i = j = 0; dist = 1.e6
    for k in pos_x:
        i += 1
        x = pos_x[k]
        y = pos_y[k]
        d = (x-x0)**2 + (y-y0)**2
        if d < dist:
            j = k
            dist = d           
    return customers[j]
    

class Customer(Agent):
    """Customer has multiple options (Cash/Card/CBDC) to exercise tansaction.
    
       She/he allocates her/his wealth and makes purchases from Merchant.

       Notation:
         w -    costs related to customer unable to buy goods (because merchant doesn't accept payment type)
         cf -   fixed cost of cash
         beta - cash_benefit
         cd -   fixed cost of credit card
         rd -   interest rate on credit card (aka benefit)
         cs -   fixed cost of cbdc
         rs -   interest rate on cbdc (aka benefit)
         x,y -  customer location
    """
    def __init__(self,model,unique_id, 
                 x,y,w,cf,cd,cs,
                 beta,rd,rs,
                 holds_CASH,holds_CD,holds_CBDC,
                 holds_CASH_CD,holds_CASH_CBDC,holds_CD_CBDC,
                 holds_CASH_CD_CBDC):
        super().__init__(unique_id, model)
        self.costs = 0
        self.x  = x
        self.y  = y
        self.w  = w
        self.cf = cf
        self.cd = cd
        self.cs = cs
        self.beta = beta
        self.rd = rd
        self.rs = rs
        self.merchant = None
        
        # Customer attributes
        self.holds_CASH = holds_CASH
        self.holds_CD = holds_CD
        self.holds_CBDC = holds_CBDC
        self.holds_CASH_CD = holds_CASH_CD
        self.holds_CASH_CBDC = holds_CASH_CBDC
        self.holds_CD_CBDC = holds_CD_CBDC
        self.holds_CASH_CD_CBDC = holds_CASH_CD_CBDC
        
        # Merchant attributes
        self.accepts_CASH = False
        self.accepts_CD = False
        self.accepts_CBDC = False
        self.accepts_CASH_CD = False
        self.accepts_CD_CBDC = False
        self.accepts_CASH_CBDC = False
        self.accepts_CASH_CD_CBDC = False
        
        
    def choosePaymentMethodAndPay(self):
        """Choose payment instrument and pay."""
        
        x,y = self.pos
        # The closest merchant
        merchant = get_closest_merchant(x,y)
        self.merchant = merchant
        merchant.w = self.w
        merchant.cf = self.cf
        merchant.cd = self.cd 
        merchant.cs = self.cs
        merchant.beta = self.beta 
        merchant.rd = self.rd
        merchant.rs = self.rs
        
        # Store customer attributes
        merchant.holds_CASH = self.holds_CASH
        merchant.holds_CD = self.holds_CD
        merchant.holds_CBDC = self.holds_CBDC
        merchant.holds_CASH_CD = self.holds_CASH_CD
        merchant.holds_CASH_CBDC = self.holds_CASH_CBDC
        merchant.holds_CD_CBDC = self.holds_CD_CBDC
        merchant.holds_CASH_CD_CBDC = self.holds_CASH_CD_CBDC
        
        # Compute merchant costs
        merchant_cost1 = merchant.sell_via_cash()
        merchant_cost2 = merchant.sell_via_creditcard()
        merchant_cost3 = merchant.sell_via_cbdc()
        merchant_cost4 = merchant.sell_via_cash_or_creditcard()
        merchant_cost5 = merchant.sell_via_cash_or_cbdc()
        merchant_cost6 = merchant.sell_via_creditcard_or_cbdc()
        merchant_cost7 = merchant.sell_via_cash_or_creditcard_or_cbdc()
        
        # All merchant costs        
        merchant_costs = np.array([merchant_cost1,merchant_cost2,merchant_cost3,merchant_cost4,merchant_cost5,merchant_cost6,merchant_cost7])
             
        # Store merchant's attributes
        self.accepts_CASH = merchant.accepts_CASH
        self.accepts_CD = merchant.accepts_CD 
        self.accepts_CBDC = merchant.accepts_CBDC
        self.accepts_CASH_CD = merchant.accepts_CASH_CD
        self.accepts_CASH_CBDC = merchant.accepts_CASH_CBDC
        self.accepts_CD_CBDC = merchant.accepts_CD_CBDC
        self.accepts_CASH_CD_CBDC = merchant.accepts_CASH_CD_CBDC
        
        # Compute customer costs
        customer_cost1 = self.pay_by_cash()
        customer_cost2 = self.pay_by_creditcard()
        customer_cost3 = self.pay_by_cbdc()
        customer_cost4 = self.pay_by_cash_or_creditcard()
        customer_cost5 = self.pay_by_cash_or_cbdc()
        customer_cost6 = self.pay_by_creditcard_or_cbdc()
        customer_cost7 = self.pay_by_cash_or_creditcard_or_cbdc()
        
        # All customer costs
        customer_costs = np.array([customer_cost1,customer_cost2,customer_cost3,customer_cost4,customer_cost5,customer_cost6,customer_cost7])
        
        # TODO - check if product or sum or any other function should be taken
        # All costs
        costs = np.kron(customer_costs,merchant_costs)
        
        # Find minimum customer cost
        ind = np.argmin(costs)
        ind = ind % n_states
        self.costs += customer_costs[ind]
        self.holds_CASH = self.holds_CD = self.holds_CBDC = self.holds_CASH_CD = False
        self.holds_CD_CBDC = self.holds_CASH_CBDC = self.holds_CASH_CD_CBDC = False
        if (ind == 1) :
           self.holds_CASH = True
        elif (ind == 2) :
           self.holds_CD = True
        elif (ind == 3) :
           self.holds_CBDC = True
        elif (ind == 4) :
           self.holds_CASH_CD = True
           self.holds_CASH  = True
           self.holds_CD = True
        elif (ind == 5) :
           self.holds_CASH_CBDC = True
           self.holds_CASH = True
           self.holds_CBDC = True
        elif (ind == 6) :
           self.holds_CD_CBDC = True
           self.holds_CD = True
           self.holds_CBDC = True
        elif (ind == 7) :
           self.holds_CASH_CD_CBDC = True
           self.holds_CASH_CD = True
           self.holds_CASH_CBDC = True
           self.holds_CD_CBDC = True
           self.holds_CASH  = True
           self.holds_CBDC = True
           self.holds_CD = True
           
        ind = ind//n_states
        merchant.costs += merchant_costs[ind]
         
    def pay_by_cash(self):
        """Consumers pay for goods by cash."""
        cost = self.cf - self.beta * self.accepts_CASH + self.w * (self.accepts_CD or self.accepts_CBDC or self.accepts_CD_CBDC)
        return cost
    
    def pay_by_creditcard(self):
        """Consumers pay for goods by credit card."""
        cost = self.cd - self.rd * self.accepts_CD + self.w * (self.accepts_CASH or self.accepts_CBDC or self.accepts_CASH_CBDC)
        return cost
    
    def pay_by_cbdc(self):
        """Consumers pay for goods by CBDC."""
        cost = self.cs - self.rs * self.accepts_CBDC + self.w * (self.accepts_CASH or self.accepts_CD or self.accepts_CASH_CD)
        return cost
    
    def pay_by_cash_or_creditcard(self):
        """Consumers pay for goods by cash or credit card."""
        cost = self.cf - self.beta * (self.accepts_CASH or self.accepts_CASH_CBDC) \
             + self.cd - self.rd * (self.accepts_CD or self.accepts_CD_CBDC) \
             - max(self.beta,self.rd) * (self.accepts_CASH_CD or self.accepts_CASH_CD_CBDC) \
             + self.w * self.accepts_CBDC
        return cost
    
    def pay_by_cash_or_cbdc(self):
        """Consumers pay for goods by cash or CBDC."""
        cost = self.cf - self.beta * (self.accepts_CASH or self.accepts_CASH_CBDC) \
             + self.cs - self.rs * (self.accepts_CBDC or self.accepts_CD_CBDC) \
             - max(self.rd,self.rs) * (self.accepts_CASH_CBDC or self.accepts_CASH_CD_CBDC) \
             + self.w * self.accepts_CD
        return cost
    
    def pay_by_creditcard_or_cbdc(self):
        """Consumers pay for goods by credit card or CBDC."""
        cost = self.cd - self.rd * (self.accepts_CD or self.accepts_CD_CBDC) \
             + self.cs - self.rs * (self.accepts_CBDC or self.accepts_CASH_CBDC) \
             - max(self.beta,self.rd) * (self.accepts_CD_CBDC or self.accepts_CASH_CD_CBDC) \
             + self.w * self.accepts_CBDC
        return cost
    
    def pay_by_cash_or_creditcard_or_cbdc(self):
        """Consumers pay for goods by cash or credit card or CBDC."""
        cost = self.cf - self.beta * self.accepts_CASH  \
             + self.cd - self.rd * self.accepts_CD    \
             + self.cs - self.rs * self.accepts_CBDC  \
             - max(self.beta,self.rd) * self.accepts_CASH_CD    \
             - max(self.beta,self.rs) * self.accepts_CASH_CBDC  \
             - max(self.rd,self.rs) * self.accepts_CD_CBDC    \
             - max(self.beta,self.rd,self.rs) * self.accepts_CASH_CD_CBDC 
        return cost
    
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,moore=True,include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
    def step(self):
        self.move()
        self.choosePaymentMethodAndPay()
        

class Merchant(Agent):
    """Merchant accepts payments from Consumers and exchanges it for goods.

       Notation:
         pc    - fixed cost of accepting cash 
         qc    - variable cost of accepting cash 
         pd    - fixed cost of accepting credit cards 
         qd    - variable cost of accepting credit cards 
         ps    - fixed cost of accepting CBDC 
         qs    - variable cost of accepting CBDC  
         ksi   - cost of missed sale 
         x,y   - merchant location
         z_cd  - share of customers that use cash rather than cards when given both instruments choice
         z_cs  - share of customers that use cash rather than cdbc when givne both instruments choice
         z_ds  - share of customers that use cards rather than cdbc when given both instruments choice
         z_cds - share of customers that use cash rather than cards or cbdc when given all instruments choice
         z_dcs - share of customers that use cards rather than cash or cdbc when given all instruments choice
         z_scd - share of customers that use cdbc rather than cash or cards when given all instruments choice
    """    
    
    def __init__(self,model,unique_id,x,y,
                 pc,qc,pd,qd,ps,qs,ksi,
                 z_cd,z_cs,z_ds,z_cds,z_dcs,z_scd,
                 accepts_CASH,accepts_CD,accepts_CBDC,
                 accepts_CASH_CD,accepts_CASH_CBDC,
                 accepts_CD_CBDC,accepts_CASH_CD_CBDC):
        super().__init__(unique_id, model)
                      
        self.costs = 0
        self.x = x
        self.y = y
        
        # Merchant costs
        self.pc = pc 
        self.qc = qc
        self.pd = pd 
        self.qd = qd  
        self.ps = ps 
        self.qs = qs 
        self.ksi = ksi
        
        # Merchant attributes
        self.accepts_CASH = accepts_CASH
        self.accepts_CD = accepts_CD 
        self.accepts_CBDC = accepts_CBDC
        self.accepts_CASH_CD  = accepts_CASH_CD
        self.accepts_CASH_CBDC = accepts_CASH_CBDC
        self.accepts_CD_CBDC = accepts_CD_CBDC
        self.accepts_CASH_CD_CBDC = accepts_CASH_CD_CBDC
        self.z_cd = z_cd
        self.z_cs = z_cs
        self.z_ds = z_ds
        self.z_scd = z_scd
        self.z_cds = z_cds
        self.z_dcs = z_dcs
        
        # Customer attributes
        self.holds_CASH = False
        self.holds_CD = False
        self.holds_CBDC = False
        self.holds_CASH_CD = False
        self.holds_CASH_CBDC = False
        self.holds_CD_CBDC = False
        self.holds_CASH_CD_CBDC = False
    
    def sell_via_cash(self):
        """Merchant accepts cash payment. """
        cost = self.pc + self.qc * self.holds_CASH + self.w * self.ksi * (self.holds_CD or self.holds_CBDC or self.holds_CD_CBDC)
        return cost
    
    def sell_via_creditcard(self):
        """Merchant accepts credit card payment. """
        cost = self.pd + self.qd * self.holds_CD + self.w * self.ksi * (self.holds_CASH or self.holds_CBDC or self.holds_CASH_CBDC)
        return cost
    
    def sell_via_cbdc(self):
        """Merchant accepts CBDC payment. """
        cost = self.ps + self.qs * self.holds_CBDC + self.w * self.ksi * (self.holds_CASH or self.holds_CD or self.holds_CASH_CD)
        return cost

    def sell_via_cash_or_creditcard(self):
        """Merchant accepts cash or credit card payment. """
        cost = self.pc + self.qc * (self.holds_CASH + self.holds_CASH_CBDC + self.z_cd*self.holds_CASH_CD) \
             + self.pd + self.qd * (self.holds_CD + self.holds_CD_CBDC + (1-self.z_cd)*self.holds_CASH_CD) \
             + self.w * self.ksi * self.holds_CBDC
        return cost
    
    def sell_via_cash_or_cbdc(self):
        """Merchant accepts credit card or CBDC payment. """
        cost = self.pc + self.qc * (self.holds_CASH + self.holds_CASH_CD + self.z_cs*self.holds_CASH_CBDC) \
             + self.ps + self.qs * (self.holds_CBDC + self.holds_CD_CBDC + (1-self.z_cs)*self.holds_CASH_CBDC) \
             + self.w * self.ksi * self.holds_CD
        return cost
    
    def sell_via_creditcard_or_cbdc(self):
        """Merchant accepts credit card or CBDC payment. """
        cost = self.pd + self.qd * (self.holds_CD  +  self.holds_CASH_CD + self.z_ds*self.holds_CD_CBDC) \
             + self.ps + self.qs * (self.holds_CBDC + self.holds_CASH_CBDC + (1-self.z_cd)*self.holds_CD_CBDC) \
             + self.w * self.ksi * self.holds_CASH
        return cost  
    
    def sell_via_cash_or_creditcard_or_cbdc(self):
        """Merchant accepts cash or credit card or CBDC payment. """
        cost = self.pc + self.pd + self.ps \
             + self.qc * (self.holds_CASH + self.z_cd*self.holds_CASH_CD + self.z_cs*self.holds_CASH_CBDC + self.z_cds*self.holds_CASH_CD_CBDC) \
             + self.qd * (self.holds_CD + (1-self.z_cd)*self.holds_CASH_CD + self.z_ds*self.holds_CD_CBDC + self.z_dcs*self.holds_CASH_CD_CBDC) \
             + self.qs * (self.holds_CBDC + (1-self.z_cs)*self.holds_CASH_CBDC + (1-self.z_ds)*self.holds_CD_CBDC + self.z_scd*self.holds_CASH_CD_CBDC) \
             + self.w * self.ksi * self.holds_CASH_CD_CBDC
        return cost
    

class Engine(Model):
    """Payment tansactions model with number of agents."""

    def __init__(self,n_customers,n_merchants,
                 # Customers parameters
                 w,cf,cd,cs,beta,rd,rs,
                 # Merchants parameters
                 z_cd,z_cs,z_ds,z_cds,z_dcs,z_scd,
                 pc,qc,pd,qd,ps,qs,ksi
                 ):
        global pos_x, pos_y, customers
        
        self.n_customers = n_customers
        self.n_merchants = n_merchants
        self.grid = MultiGrid(n_customers,n_customers,True)
        self.schedule = RandomActivation(self)
        self.running = True
        
        # Customers
        n_states = len(States)
        max_states = 2**(1+n_states)
        rnd = randint(1,max_states)
        holds_CASH,holds_CD,holds_CBDC,holds_CASH_CD,holds_CASH_CBDC, \
        holds_CD_CBDC,holds_CASH_CD_CBDC = get_states(rnd,n_states)
               
        # Merchants
        rnd = randint(1,max_states)
        accepts_CASH,accepts_CD,accepts_CBDC,accepts_CASH_CD,accepts_CASH_CBDC, \
        accepts_CD_CBDC,accepts_CASH_CD_CBDC = get_states(rnd,n_states)
        
        # Create customer agents
        for i in range(self.n_customers):
            x = self.random.randrange(n_customers)
            y = self.random.randrange(n_customers)
            a = Customer(self,i,x,y,
                         w,cf,cd,cs,
                         beta,rd,rs,
                         holds_CASH,holds_CD,holds_CBDC,
                         holds_CASH_CD,holds_CASH_CBDC,holds_CD_CBDC,
                         holds_CASH_CD_CBDC)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            self.grid.place_agent(a, (x, y))
            customers[i] = a
            
        # Create merchants agents
        for i in range(self.n_customers,self.n_customers+self.n_merchants):
            x = uniform(0,n_customers)
            y = uniform(0,n_customers)
            a = Merchant(self,i,x,y,pc,qc,pd,qd,ps,qs,ksi,
                          z_cd,z_cs,z_ds,z_cds,z_dcs,z_scd,
                          accepts_CASH,accepts_CD,accepts_CBDC,
                          accepts_CASH_CD,accepts_CASH_CBDC,
                          accepts_CD_CBDC,accepts_CASH_CD_CBDC)
            self.schedule.add(a)
            # Save agent location
            pos_x[i] = x
            pos_y[i] = y
            customers[i] = a
            
        self.datacollector = DataCollector(
            model_reporters={"CustomerStatistics": compute_customer_statistics, 
                             "MerchantStatistics": compute_merchant_statistics}, 
            agent_reporters={"Costs": "costs"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':

    ### Iteration process to compute the final equilibrium reached from each starting point on the lattice  
    ### Explore a lattice of possible values of two parameters
    # 	pc between 0 and 1;
    # 	pd between 0  and 1;
    # 	rd = 0.5;
    # 	w = 1;
    # 	qc between 0 and 1;
    # 	qd between 0 and 1;
    # 	ksi = 1;
    #   ps=0.5
    #   qs=0.5

    horizon = 5; 
    n_customers=100; n_merchants=n_customers//10 # every merchant for 100 customers
    # Customer's parameters
    w=1; cf=0.3; cd=0.3; cs=0.3; beta=0.5; rd=0.05; rs=0.01
    # Merchant's parameters
    pc=0.3; qc=0.5; pd=0.3; qd=0.5; ps=0.5; qs=0.5; ksi=1
    z_cd=1.0; z_cs=1.0; z_ds=1.0; z_cds=0.3; z_dcs=0.5; z_scd=1.0-z_cds-z_dcs
    n1 = n_customers; n2 = n1 + n_merchants
    
    # Create a model with 112 agents
    model = Engine(n_customers=100,n_merchants=10,
                   # Customers parameters
                   w=w,cf=cf,cd=cd,cs=cs,beta=beta,rd=rd,rs=rs,
                   # Merchants parameters
                   z_cd=z_cd,z_cs=z_cs,z_ds=z_ds,z_cds=z_cds,z_dcs=z_dcs,z_scd=z_scd,
                   pc=pc,qc=qc,pd=pd,qd=qd,ps=ps,qs=qs,ksi=ksi)
    
    for i in range(horizon):
        model.step()
        
    agents = model.schedule.agents
    customers_costs = [a.costs for a in agents if isinstance(a,Customer)]
    merchants_costs = [a.costs for a in agents if isinstance(a,Merchant)]
    cash = [a.holds_CASH for a in agents if isinstance(a,Customer)]
    cards = [a.holds_CD for a in agents if isinstance(a,Customer)]
    cbdc = [a.holds_CBDC for a in agents if isinstance(a,Customer)]
    cash_cd = [a.holds_CASH_CD for a in agents if isinstance(a,Customer)]
    cash_cbdc = [a.holds_CASH_CBDC for a in agents if isinstance(a,Customer)]
    cd_cbdc = [a.holds_CD_CBDC for a in agents if isinstance(a,Customer)]
    cash_cd_cbdc = [a.holds_CASH_CD_CBDC for a in agents if isinstance(a,Customer)]
    
    # Visualize the number of agents residing in each cell
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, (x, y) = cell
        agent_count = len(cell_content)
        if agent_count > 0:
            agent_counts[x][y] = agent_count
    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()
    plt.savefig(os.path.join(path,"../../../graphs/CBDC/grid.png"),dpi=600)
    plt.show()
    
    plt.rcParams.update({'font.size': 7}) # must set in top
    # Plot statistics
    fig = plt.figure(figsize=(8,4))
    stat = model.datacollector.get_model_vars_dataframe()
    statistics = []
    for i in range(horizon):
        statistics.append(stat.iloc[i].values[0])
    statistics = pandas.DataFrame(statistics,columns=states)
    #statistics.plot(subplots=True, ax=axes)
    #statistics.plot(kind="line",xlabel="Period",title="Share of Payment Instruments",grid=True)
    for i,col in enumerate(statistics.columns):
        ax = plt.subplot(2,4,1+i)
        ax.plot(statistics[col],label=col,linewidth=2)
        plt.title(col)
        plt.grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(path,"../../../graphs/CBDC/grid.png"))
    plt.show()
    
    plt.rcParams.update({'font.size': 15})
    # Display agent costs data
    agent_costs = model.datacollector.get_agent_vars_dataframe()
    temp = np.array(agent_costs.values).reshape((n2,len(agent_costs)//n2))
    agent_costs = pandas.DataFrame(temp)
    #print(agent_costs.head())
    
    # # Plot histogram of agent costs at the model’s end
    # fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(8,12))
    # customer_end_costs = agent_costs[:n1]
    # customer_end_costs.plot(kind='hist',title="Customer Costs",grid=True,xlabel="Period",ylabel="Costs",ax=axes[0])
    # merchant_end_costs = agent_costs[n1:n2]
    # merchant_end_costs.plot(kind='hist',title="Merchant Costs",grid=True,xlabel="Period",ylabel="Costs",ax=axes[1])       
    # fig.savefig(os.path.join(path,"../../../graphs/CBDC/cost_distribution.png"))
    # plt.show()

    # # Plot the costs of a given agent (in this example, agent 10):
    # one_agent_costs = agent_costs.iloc[10]
    # one_agent_costs.plot(title="Agents Costs",grid=True,xlabel="Period",ylabel="Costs")
    # plt.show()
    
    # customer_legend = []; merchant_legend = []
    # fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    # for i in range(horizon):
    #     j = randint(1,n1-1)
    #     customer_legend.append(f"{1+j}")
    #     one_agent_costs = agent_costs.iloc[j]
    #     one_agent_costs.plot(title="Customers Costs",xlabel=None,grid=True,legend=f"customer {1+j}")
    # plt.legend(customer_legend)
    # plt.xlim(0,horizon-1)    
    # fig.savefig(os.path.join(path,"../../../graphs/CBDC/customer_costs.png"))
    # plt.show()
    
    # fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    # for i in range(Nhorizon):
    #     j = randint(n1,n2-1)
    #     merchant_legend.append(f"{1+j}")
    #     one_agent_costs = agent_costs.iloc[j]
    #     one_agent_costs.plot(title="Merchants Costs",xlabel="Period",grid=True,legend=f"merchant {1+j}")
    # plt.legend(merchant_legend)
    # plt.xlim(0,horizon-1)
    # fig.savefig(os.path.join(path,"../../../graphs/CBDC/merchant_costs.png"))
    # plt.show()
    
    # Save the agent data (stored in the pandas agent_costs object) to CSV
    # agent_costs.to_csv("agent_data.csv")

    ############################################################ Batch Run
    Niterations = 1
    # Merchant's parameters
    qc=np.arange(0,1,0.1) 
    qd=np.arange(0,1,0.1)
    qs=np.arange(0,1,0.5) 
    ksi = 0.5
    params = { # Common parameters
               "n_customers": 100, "n_merchants": 10,
               # Customers parameters
               "w": w, "cf": cf, "cd": cd, "cs": cs,
               "beta": beta, "rd": rd, "rs": rs,
               # Merchants parameters
               "z_cd": z_cd, "z_cs": z_cs, "z_ds": z_ds,
               "z_cds": z_cds, "z_dcs": z_dcs, "z_scd": z_scd,
               "pc": pc, "qc": qc, "pd": pd, "qd": qd,
               "ps": ps, "qs": qs, "ksi": ksi
        }
    
    results = batch_run(
        Engine,
        parameters=params,
        iterations=Niterations,
        max_steps=horizon,
        number_processes=5,
        data_collection_period=1,
        display_progress=True,
    )

    results_df = pandas.DataFrame(results)
    print(results_df.keys())
    
    # Filter our results to only contain the data of one agent at the end of each episode
    fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    results_filtered = results_df[(results_df.AgentID == 0) & (results_df.Step == horizon)]
    customer_statistics = results_filtered.CustomerStatistics.values
    for i in range(len(customer_statistics)):
        plt.scatter(range(n_states),customer_statistics[i])
    tics = [x.lower().replace("_and_","-").replace("_","") for x in states]
    axes.set_xticklabels(labels=[""]+tics,rotation=90,ha='left')
    plt.ylabel('Percentage')
    plt.title('Payment Instruments of the First Customer')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Filter our results to only contain the data at the last steps of each episode
    cash=[];cd=[];cdbc=[];cash_cd=[];cash_cdbc=[];cd_cdbc=[];cash_cd_cdbc=[];pic=[];pid=[]
    for i in range(n_customers):
        cust = customers[i]
        merch = cust.merchant
        results_filtered = results_df[(results_df.AgentID == i) & (results_df.Step == horizon)]
        # Customers
        customer_statistics = results_filtered.CustomerStatistics.values
        for st in customer_statistics:
            h_cash,h_cd,h_cdbc,h_cash_cd,h_cash_cdbc,h_cd_cdbc,h_cash_cd_cdbc = st
            cash.append(h_cash);cd.append(h_cd);cdbc.append(h_cdbc);cash_cd.append(h_cash_cd) 
            cash_cdbc.append(h_cash_cdbc);cd_cdbc.append(h_cd_cdbc);cash_cd_cdbc.append(h_cash_cd_cdbc)
        # Merchants
        merchant_statistics = results_filtered.MerchantStatistics.values
        for st in merchant_statistics:
            a_cash,a_cd,a_cdbc,a_cash_cd,a_cash_cdbc,a_cd_cdbc,a_cash_cd_cdbc = st
            pic.append(a_cash+a_cash_cd+a_cash_cd_cdbc+a_cash_cdbc)
            pid.append(a_cd+a_cash_cd+a_cash_cd_cdbc+a_cd_cdbc)
    
    
    Mx   = np.zeros((n_customers,n_customers))
    My   = np.zeros((n_customers,n_customers))
    Mz   = np.zeros((n_customers,n_customers))
    
    for i,pc in enumerate(pic):
        pc = int((n_customers-1)*pc)
        pd = int((n_customers-1)*pid[i])
        Mx[pc,pd] += cash[i]
        My[pc,pd] += cd[i]
        Mz[pc,pd] += cash_cd[i]
           
    #fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    plt.matshow(Mx,cmap=plt.cm.viridis)
    plt.colorbar()
    plt.title("Cash")
    plt.show()
    
    #fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    plt.matshow(My,cmap=plt.cm.viridis)
    plt.colorbar()
    plt.title("Credit Card")
    plt.show()
    
    #fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    plt.matshow(Mz,cmap=plt.cm.viridis)
    plt.colorbar()
    plt.title("Cash and Credit Card")
    plt.show()
    
    costs = results_filtered.Costs.values
    plt.plot(costs)
    plt.title('Customers Cost')
    plt.grid(True)
    plt.savefig(os.path.join(path,"../../../graphs/CBDC/customers_cost.png"))
    plt.show()
    
    # Display the agent’s costs at each time step of one specific episode
    one_episode_costs = results_df[results_df.iteration == 0]
    # Then, print the columns of interest of the filtered data frame
    print()
    print(one_episode_costs.to_string(index=False, columns=["Step","AgentID","Costs"],max_rows=25)
        
)