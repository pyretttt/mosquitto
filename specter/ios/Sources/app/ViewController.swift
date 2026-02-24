import UIKit
import SwiftUI
import SnapKit
import ios_Base

class ViewController: UIViewController {
    
    let moduleService: ModuleService
    
    override init(nibName nibNameOrNil: String?, bundle nibBundleOrNil: Bundle?) {
        weak var weakSelf: ViewController?
        self.moduleService = ModuleService(modules: allModules) { module in
            let moduleViewController = modify(ModuleDetailViewController()) {
                $0.modalPresentationStyle = .fullScreen
            }
            weakSelf?.present(moduleViewController, animated: true)
        }

        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
        weakSelf = self
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let hostingController = UIHostingController(
            rootView: MainView(moduleService: moduleService)
        )

        addChild(hostingController)
        view.addSubview(hostingController.view)
        
        hostingController.view.snp.makeConstraints { (make) -> Void in
            make.leading.trailing.equalTo(self.view)
            make.top.bottom.equalTo(self.view)
         }
        
        hostingController.didMove(toParent: self)
    }
}
