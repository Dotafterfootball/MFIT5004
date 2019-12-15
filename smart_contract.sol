
pragma solidity ^0.4.24;

contract S_C{
    //deployed contract address at ropsten network: 0xb3C5945dB08af8779A68C25e87d24F7DA4F71d4b
    address owner;
    mapping(address => uint256) balances;
    mapping(address => bool) inBank; //record whether one address is in the account list
    uint256 funding;
    address[] account;
    bool bankopen = true;
    
    event AuditLog(
        address indexed _From,
        uint _Value
    );
    
    constructor() public payable{
        require(msg.value>2000000000000000000); // require initial funding of bank is larger than 2 ether
        owner = msg.sender;
        funding = msg.value;
    }
    
    function deposit() public payable{
        require(bankopen);
        if(inBank[msg.sender] != true){
             require(account.length<10); //max 10 accounts are allowed
             account.push(msg.sender);
        }
        
        inBank[msg.sender] = true;
        balances[msg.sender]+=msg.value;
        emit AuditLog(msg.sender, msg.value);
    }
    
    function withdraw(uint256 amount) public {
        require(bankopen);
        require(inBank[msg.sender]==true); //ensure the address has a account in the bank
        require(balances[msg.sender] >= amount && amount>0);
        balances[msg.sender] -= amount;
        msg.sender.transfer(amount);
        emit AuditLog(msg.sender, amount);
    }
    
    function borrow(uint256 amount) public {
        require(bankopen);
        if(inBank[msg.sender] != true){
             require(account.length<10);
             account.push(msg.sender);
             inBank[msg.sender] = true;
        }
        require((this.balance-amount)>=1000000000000000000); //Not allow to borrow if bank’s total remaining balance is < 1 ether after transfer ether to borrower
        uint interest = amount/100*5;
        balances[msg.sender] -= amount + interest;
        msg.sender.transfer(amount);
        emit AuditLog(msg.sender, amount);
    }
    
    function getRemainingBankBalance() public view returns (uint256) {
        return this.balance;
    }
    
    function colseBank() public {
        require(bankopen);
        require(msg.sender==owner); //only owner can close the bank
        for(uint i=0;i<account.length;i++){
            if(balances[account[i]]<0){
                return; //ensure no negative balance before transfer
            }
        }
        
        for(uint j=0;j<account.length;j++){
            account[j].transfer(balances[account[j]]);
            emit AuditLog(msg.sender, balances[account[j]]);
        }
        owner.transfer(funding);
        emit AuditLog(msg.sender, funding);
        bankopen=false;
    }
    function getbalance() public view returns (int256){
        require(bankopen);
        return int(balances[msg.sender]);
    }
    function getAccountLength() public view returns (uint256){
        require(bankopen);
        return account.length;
    }
    function setBankOwner(address addre) public{
        owner = addre;
    }
}
