import UIKit
import SwiftUI
import SnapKit

class ViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        let hostingController = UIHostingController(rootView: MainView())

        addChild(hostingController)
        view.addSubview(hostingController.view)
        
        hostingController.view.snp.makeConstraints { (make) -> Void in
            make.leading.trailing.equalTo(self.view)
            make.top.bottom.equalTo(self.view)
         }
        
        print("Zalupa")
        
        hostingController.didMove(toParent: self)
    }
}
