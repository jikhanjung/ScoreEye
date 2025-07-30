

import fitz  # PyMuPDF
import os
import sys

def convert_pdf_to_png(pdf_path, output_dir, dpi=300):
    """
    PDF 파일의 각 페이지를 고해상도 PNG 이미지로 변환합니다.

    :param pdf_path: 변환할 PDF 파일의 경로
    :param output_dir: PNG 파일을 저장할 디렉터리
    :param dpi: 이미지 해상도 (dots per inch)
    """
    # PDF 파일이 존재하는지 확인
    if not os.path.exists(pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다 - '{pdf_path}'")
        return

    # 출력 디렉터리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉터리를 생성했습니다: '{output_dir}'")

    try:
        # PDF 파일 열기
        doc = fitz.open(pdf_path)
        
        # PDF 파일 이름 (확장자 제외)
        pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        print(f"'{pdf_path}' 파일 변환을 시작합니다. 총 {doc.page_count} 페이지...")

        # 각 페이지를 순회하며 PNG로 변환
        for page_num in range(doc.page_count):
            # 페이지 로드
            page = doc.load_page(page_num)

            # 픽셀맵(이미지) 생성
            # DPI를 높이면 이미지 품질이 좋아지지만 파일 크기가 커집니다.
            pix = page.get_pixmap(dpi=dpi)

            # 출력 파일 경로 설정 (예: output/my_pdf_page_0001.png)
            output_file_name = f"{pdf_base_name}_page_{page_num + 1:04d}.png"
            output_file_path = os.path.join(output_dir, output_file_name)

            # PNG 파일로 저장
            pix.save(output_file_path)
            
            # 진행 상황 출력 (end='\r'은 줄바꿈 없이 현재 줄에 덮어쓰기 위함)
            print(f"  - {page_num + 1}/{doc.page_count} 페이지 저장 완료: {output_file_name}", end='\r')

        doc.close()
        print("\n변환이 완료되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    # 명령줄 인수가 올바르게 주어졌는지 확인
    if len(sys.argv) != 2:
        print("사용법: python pdf_to_png_converter.py <PDF_파일_경로>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    
    # 출력 디렉터리 이름 설정 (예: my_pdf_images)
    pdf_base_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
    output_directory = f"{pdf_base_name}_png_pages"

    convert_pdf_to_png(pdf_file_path, output_directory)

