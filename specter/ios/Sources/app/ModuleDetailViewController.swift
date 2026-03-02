import UIKit

final class ModuleDetailViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        
        let helloLabel = UILabel()
        helloLabel.textAlignment = .center
        helloLabel.numberOfLines = 0
        helloLabel.text = "\(RustBridge.hello())\nNumber: \(RustBridge.number())"
        helloLabel.translatesAutoresizingMaskIntoConstraints = false
        
        view.addSubview(helloLabel)
        
        NSLayoutConstraint.activate([
            helloLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            helloLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            helloLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 16),
            helloLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -16)
        ])
    }
}
