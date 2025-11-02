"""
MT5 Login Configuration Tool
Easy management of MetaTrader 5 connection settings for different brokers
NOW USES .ENV FILE FOR SECURE CREDENTIAL STORAGE (v1.5.0+)
"""
import json
import os
from colorama import init, Fore, Style
from dotenv import load_dotenv, set_key, find_dotenv

# Initialize colorama for Windows
init()

class MT5ConfigManager:
    def __init__(self):
        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'config.json'
        )
        self.env_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            '.env'
        )
        # Load environment variables
        load_dotenv(self.env_path)
        self.load_config()
    
    def load_config(self):
        """Load current configuration from config.json and credentials from .env"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Load MT5 credentials from .env (takes priority)
            self.config['mt5']['login'] = int(os.getenv('MT5_LOGIN', 0))
            self.config['mt5']['password'] = os.getenv('MT5_PASSWORD', '')
            self.config['mt5']['server'] = os.getenv('MT5_SERVER', '')
            
        except FileNotFoundError:
            print(f"{Fore.RED}[ERROR] Error: Config file not found at {self.config_path}{Style.RESET_ALL}")
            self.config = {}
        except KeyError:
            print(f"{Fore.YELLOW}[WARNING] Warning: 'mt5' section missing in config{Style.RESET_ALL}")
    
    def save_config(self):
        """Save credentials to .env and other settings to config.json"""
        try:
            # Ensure .env file exists
            if not os.path.exists(self.env_path):
                with open(self.env_path, 'w') as f:
                    f.write("# MT5 Credentials (v1.5.0+)\n")
            
            # Save credentials to .env file
            set_key(self.env_path, 'MT5_LOGIN', str(self.config['mt5'].get('login', 0)))
            set_key(self.env_path, 'MT5_PASSWORD', self.config['mt5'].get('password', ''))
            set_key(self.env_path, 'MT5_SERVER', self.config['mt5'].get('server', ''))
            
            # Reload environment variables after saving
            load_dotenv(self.env_path, override=True)
            
            # Clear credentials from config.json (keep other settings)
            config_copy = self.config.copy()
            config_copy['mt5']['login'] = 0
            config_copy['mt5']['password'] = ''
            config_copy['mt5']['server'] = ''
            
            with open(self.config_path, 'w') as f:
                json.dump(config_copy, f, indent=2)
            
            print(f"{Fore.GREEN}[OK] Credentials saved to .env file (secure){Style.RESET_ALL}")
            print(f"{Fore.GREEN}[OK] Other settings saved to config.json{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Error saving config: {e}{Style.RESET_ALL}")
            return False
    
    def display_current_config(self):
        """Display current MT5 login settings from .env and config.json"""
        mt5_config = self.config.get('mt5', {})
        
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{'MT5 LOGIN CONFIGURATION (v1.5.0+)':^80}{Style.RESET_ALL}")
        print("=" * 80)
        
        print(f"\n{Fore.YELLOW}CURRENT SETTINGS:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}(Credentials loaded from .env file){Style.RESET_ALL}")
        print("-" * 80)
        
        login = mt5_config.get('login', 'Not set')
        password = mt5_config.get('password', 'Not set')
        server = mt5_config.get('server', 'Not set')
        path = mt5_config.get('path', 'Not set')
        timeout = mt5_config.get('timeout', 60000)
        portable = mt5_config.get('portable', False)
        
        # Mask password for display
        masked_password = '*' * len(str(password)) if password != 'Not set' else 'Not set'
        
        print(f"  Account Number:  {Fore.GREEN}{login}{Style.RESET_ALL} {Fore.CYAN}(.env){Style.RESET_ALL}")
        print(f"  Password:        {Fore.GREEN}{masked_password}{Style.RESET_ALL} {Fore.CYAN}(.env){Style.RESET_ALL}")
        print(f"  Server:          {Fore.GREEN}{server}{Style.RESET_ALL} {Fore.CYAN}(.env){Style.RESET_ALL}")
        print(f"  MT5 Path:        {Fore.CYAN}{path}{Style.RESET_ALL} (config.json)")
        print(f"  Timeout:         {timeout} ms (config.json)")
        print(f"  Portable Mode:   {portable} (config.json)")
        
        print("\n" + "=" * 80)
    
    def update_login_settings(self):
        """Interactive update of MT5 login settings"""
        print(f"\n{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.CYAN}              UPDATE MT5 LOGIN SETTINGS                     {Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        mt5_config = self.config.get('mt5', {})
        
        print(f"\n{Fore.YELLOW}Press Enter to keep current value, or type new value:{Style.RESET_ALL}\n")
        
        # Account Number
        current_login = mt5_config.get('login', '')
        new_login = input(f"Account Number (current: {Fore.GREEN}{current_login}{Style.RESET_ALL}): ").strip()
        if new_login:
            try:
                mt5_config['login'] = int(new_login)
            except ValueError:
                print(f"{Fore.RED}Invalid account number. Keeping current value.{Style.RESET_ALL}")
        
        # Password
        current_password = mt5_config.get('password', '')
        masked_current = '*' * len(str(current_password)) if current_password else 'Not set'
        new_password = input(f"Password (current: {Fore.GREEN}{masked_current}{Style.RESET_ALL}): ").strip()
        if new_password:
            mt5_config['password'] = new_password
        
        # Server
        current_server = mt5_config.get('server', '')
        print(f"\n{Fore.CYAN}Common Servers:{Style.RESET_ALL}")
        print("  - ICMarkets-Demo")
        print("  - ICMarkets-Live01")
        print("  - Pepperstone-Demo")
        print("  - Pepperstone-Live")
        print("  - FTMO-Demo")
        print("  - FTMO-Server")
        new_server = input(f"\nServer (current: {Fore.GREEN}{current_server}{Style.RESET_ALL}): ").strip()
        if new_server:
            mt5_config['server'] = new_server
        
        # MT5 Path
        current_path = mt5_config.get('path', '')
        print(f"\n{Fore.CYAN}Default MT5 Paths:{Style.RESET_ALL}")
        print("  - C:\\Program Files\\MetaTrader 5\\terminal64.exe")
        print("  - C:\\Program Files (x86)\\ICMarkets MetaTrader 5\\terminal64.exe")
        new_path = input(f"\nMT5 Path (current: {Fore.CYAN}{current_path}{Style.RESET_ALL}): ").strip()
        if new_path:
            mt5_config['path'] = new_path
        
        # Timeout
        current_timeout = mt5_config.get('timeout', 60000)
        new_timeout = input(f"\nConnection Timeout in ms (current: {current_timeout}): ").strip()
        if new_timeout:
            try:
                mt5_config['timeout'] = int(new_timeout)
            except ValueError:
                print(f"{Fore.RED}Invalid timeout. Keeping current value.{Style.RESET_ALL}")
        
        # Update config
        self.config['mt5'] = mt5_config
        
        # Confirm and save
        print(f"\n{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}                   CONFIRM CHANGES                          {Style.RESET_ALL}")
        print(f"{Fore.YELLOW}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        print(f"\nAccount Number:  {Fore.GREEN}{mt5_config.get('login', 'Not set')}{Style.RESET_ALL}")
        print(f"Password:        {Fore.GREEN}{'*' * len(str(mt5_config.get('password', '')))}{Style.RESET_ALL}")
        print(f"Server:          {Fore.GREEN}{mt5_config.get('server', 'Not set')}{Style.RESET_ALL}")
        print(f"MT5 Path:        {Fore.CYAN}{mt5_config.get('path', 'Not set')}{Style.RESET_ALL}")
        
        confirm = input(f"\n{Fore.GREEN}Save these settings? (y/n): {Style.RESET_ALL}").strip().lower()
        
        if confirm == 'y':
            if self.save_config():
                print(f"\n{Fore.GREEN}[OK] MT5 login settings saved successfully!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}[WARNING] Remember to restart the trading system for changes to take effect.{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}[ERROR] Failed to save settings.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}[INFO] Changes discarded.{Style.RESET_ALL}")
    
    def quick_broker_setup(self):
        """Quick setup for common brokers"""
        print(f"\n{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        print(f"{Fore.CYAN}              QUICK BROKER SETUP                            {Style.RESET_ALL}")
        print(f"{Fore.CYAN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        brokers = {
            '1': {
                'name': 'IC Markets Demo',
                'server': 'ICMarkets-Demo',
                'path': 'C:\\Program Files (x86)\\ICMarkets MetaTrader 5\\terminal64.exe'
            },
            '2': {
                'name': 'IC Markets Live',
                'server': 'ICMarkets-Live01',
                'path': 'C:\\Program Files (x86)\\ICMarkets MetaTrader 5\\terminal64.exe'
            },
            '3': {
                'name': 'Pepperstone Demo',
                'server': 'Pepperstone-Demo',
                'path': 'C:\\Program Files\\Pepperstone MetaTrader 5\\terminal64.exe'
            },
            '4': {
                'name': 'Pepperstone Live',
                'server': 'Pepperstone-Live',
                'path': 'C:\\Program Files\\Pepperstone MetaTrader 5\\terminal64.exe'
            },
            '5': {
                'name': 'FTMO Demo',
                'server': 'FTMO-Demo',
                'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
            },
            '6': {
                'name': 'FTMO Live',
                'server': 'FTMO-Server',
                'path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
            }
        }
        
        print(f"\n{Fore.YELLOW}SELECT BROKER:{Style.RESET_ALL}\n")
        for key, broker in brokers.items():
            print(f"  {key}. {broker['name']}")
        print(f"  0. Custom/Other")
        
        choice = input(f"\n{Fore.GREEN}Enter choice (0-6): {Style.RESET_ALL}").strip()
        
        if choice == '0':
            self.update_login_settings()
            return
        
        if choice not in brokers:
            print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")
            return
        
        broker = brokers[choice]
        
        print(f"\n{Fore.CYAN}Setting up: {broker['name']}{Style.RESET_ALL}\n")
        
        # Get account details
        login = input(f"Enter Account Number: ").strip()
        password = input(f"Enter Password: ").strip()
        
        if not login or not password:
            print(f"{Fore.RED}Account number and password are required.{Style.RESET_ALL}")
            return
        
        # Check if custom path is needed
        custom_path = input(f"\nUse custom MT5 path? (y/n, default: n): ").strip().lower()
        if custom_path == 'y':
            mt5_path = input(f"Enter MT5 Path: ").strip()
        else:
            mt5_path = broker['path']
        
        # Update config
        mt5_config = self.config.get('mt5', {})
        mt5_config['login'] = int(login)
        mt5_config['password'] = password
        mt5_config['server'] = broker['server']
        mt5_config['path'] = mt5_path
        
        self.config['mt5'] = mt5_config
        
        # Save
        if self.save_config():
            print(f"\n{Fore.GREEN}[OK] {broker['name']} configured successfully!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[WARNING] Remember to restart the trading system for changes to take effect.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}[ERROR] Failed to save settings.{Style.RESET_ALL}")
    
    def test_connection(self):
        """Test MT5 connection with current settings"""
        print(f"\n{Fore.CYAN}Testing MT5 connection...{Style.RESET_ALL}")
        
        try:
            import MetaTrader5 as mt5
            
            mt5_config = self.config.get('mt5', {})
            login = mt5_config.get('login')
            password = mt5_config.get('password')
            server = mt5_config.get('server')
            path = mt5_config.get('path')
            timeout = mt5_config.get('timeout', 60000)
            
            # Initialize MT5
            if not mt5.initialize(path=path, login=login, password=password, server=server, timeout=timeout):
                error_code = mt5.last_error()
                print(f"{Fore.RED}[ERROR] Connection failed: {error_code}{Style.RESET_ALL}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                print(f"{Fore.RED}[ERROR] Failed to get account info{Style.RESET_ALL}")
                mt5.shutdown()
                return False
            
            print(f"\n{Fore.GREEN}[OK] Connection successful!{Style.RESET_ALL}\n")
            print(f"Account: {account_info.login}")
            print(f"Server: {account_info.server}")
            print(f"Balance: ${account_info.balance:,.2f}")
            print(f"Equity: ${account_info.equity:,.2f}")
            print(f"Margin: ${account_info.margin:,.2f}")
            print(f"Free Margin: ${account_info.margin_free:,.2f}")
            
            mt5.shutdown()
            return True
            
        except ImportError:
            print(f"{Fore.RED}[ERROR] MetaTrader5 module not installed{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Error: {e}{Style.RESET_ALL}")
            return False
    
    def run(self):
        """Main menu loop"""
        while True:
            self.display_current_config()
            
            print(f"\n{Fore.CYAN}OPTIONS:{Style.RESET_ALL}")
            print("  1. Update login settings (manual)")
            print("  2. Quick broker setup (preset)")
            print("  3. Test connection")
            print("  4. Exit")
            
            choice = input(f"\n{Fore.GREEN}Enter choice (1-4): {Style.RESET_ALL}").strip()
            
            if choice == '1':
                self.update_login_settings()
            elif choice == '2':
                self.quick_broker_setup()
            elif choice == '3':
                self.test_connection()
                input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            elif choice == '4':
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")


def main():
    """Main entry point"""
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║           MT5 LOGIN CONFIGURATION TOOL                    ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    manager = MT5ConfigManager()
    manager.run()
    
    print(f"\n{Fore.CYAN}Configuration tool closed.{Style.RESET_ALL}\n")


if __name__ == '__main__':
    main()
