import UIKit
import SwiftUI
import SnapKit

class ViewController: UIViewController {
    
    let moduleService = ModuleService()
    
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
